# nano—vLLM分析

## 目录

1. 从一个朴素的推理循环说起
2. 请求调度器的设计
3. 页式 KV 缓存
4. 模型执行后端
5. 注意力机制的工程实现
6. 张量并行
7. 模型层实现
8. 采样与终止
9. 串联与验证
10. 总结



------

## 从一个朴素的推理循环说起

大语言模型的推理过程，本质上就是一个循环：给定一段输入 tokens，模型输出下一个 token 的概率分布，采样得到一个 token，把它追加到输入序列，然后重复这个过程，直到生成结束。

一个最朴素的实现大概长这样：

```python
def generate(model, input_ids, max_tokens):
    for _ in range(max_tokens):
        logits = model(input_ids)          # 前向传播
        next_token = sample(logits[:, -1]) # 只取最后一个位置的 logits
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token == eos_token:
            break
    return input_ids
```

这个实现能跑，但跑不快。问题在哪里？

每次循环调用 `model(input_ids)` 时，模型会对整个序列做一次完整的前向传播。假设序列长度是 1000，那么 Attention 层要计算 1000 个 query 对 1000 个 key 的注意力分数。但实际上，前 999 个位置的计算结果和上一轮是一样的，只有最后一个位置是新的。

这就引出了 LLM 推理的第一个核心优化：KV Cache。

### Prefill 和 Decode 的本质区别

引入 KV Cache 之后，推理过程自然地分成了两个阶段：

**Prefill 阶段**：处理完整的 prompt。所有 token 并行计算，生成初始的 KV Cache。这个阶段是计算密集型的，GPU 的算力能被充分利用。

**Decode 阶段**：逐个生成新 token。每次只有 1 个 query，但要和历史所有 key 做 attention。这个阶段是访存密集型的，瓶颈在于从显存读取 KV Cache。

来看一组实际的数据对比：

| 阶段    | 输入规模   | 计算复杂度 | 访存量   | 瓶颈 |
| ------- | ---------- | ---------- | -------- | ---- |
| Prefill | N 个 token | O(N² × d)  | O(N × d) | 计算 |
| Decode  | 1 个 token | O(N × d)   | O(N × d) | 访存 |

Prefill 的计算量和序列长度平方成正比，但可以高度并行；Decode 每步计算量小，但每步都要把整个 KV Cache 从显存搬到计算单元，显存带宽成为瓶颈。

### 为什么朴素实现跑不快

单请求的 KV Cache 优化只是第一步。当我们要服务多个用户时，新的问题出现了：

**GPU 利用率低**

Decode 阶段每个请求每次只算 1 个 token，一张卡同时只服务一个请求的话，大部分算力都浪费了。我们需要把多个请求组成一个 batch 一起算。

**Batch 难以组织**

不同请求的 prompt 长度不一样，生成长度也不一样。有的请求已经生成完了，有的还在继续。传统的静态 batch 要等所有请求都结束才能开始下一批，这期间已经结束的请求只能干等着。

**显存预分配浪费**

生成长度是不确定的，如果按照最大可能长度预分配 KV Cache，显存会被严重浪费。假设最大长度是 4096，但实际平均生成 200 个 token，那 95% 的预分配显存都是浪费的。

这三个问题指向了同一个方向：我们需要一个能够动态管理请求、动态分配显存、持续组织 batch 的调度系统。

### 我们要实现什么

这个系列要实现的推理框架包含以下核心组件：

| 组件         | 职责                                  |
| ------------ | ------------------------------------- |
| LLMEngine    | 主循环，协调调度和执行                |
| Scheduler    | 决定每一步执行哪些请求                |
| BlockManager | 页式 KV 缓存的分配与回收              |
| ModelRunner  | 准备输入、执行模型、处理输出          |
| Attention    | 集成 Flash Attention 和 KV Cache 写入 |
| 并行层       | 支持张量并行的线性层和 Embedding      |

整体数据流是这样的：

```text
用户请求 → LLMEngine.add_request() → Scheduler.waiting 队列
                                          ↓
主循环 → Scheduler.schedule() → 选出本步要执行的请求
                                          ↓
       → ModelRunner.run() → 模型前向 + 采样
                                          ↓
       → Scheduler.postprocess() → 更新状态，终止判断
                                          ↓
       → 输出完成的请求
```

下面我们从调度器开始，一步步把这个框架搭起来。

------

## 请求调度器的设计

调度器要解决的核心问题是：每一步应该执行哪些请求？

这个问题比看起来要复杂。我们有两种不同性质的操作（Prefill 和 Decode），有限的资源（显存、计算带宽），以及不断变化的请求状态。调度器需要在这些约束下，让系统吞吐量最大化，同时保证每个请求都能取得进展。

### 请求的生命周期

先定义请求的状态。一个请求从进入系统到完成，会经历三个状态：

```python
class SequenceStatus(Enum):
    WAITING = auto()   # 等待 Prefill
    RUNNING = auto()   # 正在生成
    FINISHED = auto()  # 生成完成
```

新请求进来时是 WAITING 状态，完成 Prefill 后变成 RUNNING，生成结束后变成 FINISHED。

请求的数据结构需要记录哪些信息？

```python
class Sequence:
    def __init__(self, token_ids, sampling_params):
        self.seq_id = next(Sequence.counter)  # 唯一标识
        self.status = SequenceStatus.WAITING
        self.token_ids = token_ids            # 当前所有 token
        self.last_token = token_ids[-1]       # 最后一个 token（Decode 用）
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0            # Prefix Cache 命中的 token 数
        self.block_table = []                 # 分配的物理块列表
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

几个要注意的点：

1. `last_token` 单独存一份，Decode 阶段只需要这一个 token 作为输入，不用每次都去 `token_ids[-1]`
2. `num_cached_tokens` 用于 Prefix Cache，后面会详细讲
3. `block_table` 记录这个请求占用了哪些物理内存块

还需要一些计算属性：

```python
@property
def num_blocks(self):
    return (len(self) + block_size - 1) // block_size

@property
def last_block_num_tokens(self):
    return len(self) - (self.num_blocks - 1) * block_size
```

### 调度器的核心结构

调度器维护两个队列：

```python
class Scheduler:
    def __init__(self, config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.block_manager = BlockManager(...)
        self.waiting = deque()  # WAITING 状态的请求
        self.running = deque()  # RUNNING 状态的请求
```

`max_num_seqs` 限制同时处理的最大请求数，`max_num_batched_tokens` 限制一步最多处理的 token 数。这两个参数控制了每步的 batch 大小。

### Prefill 调度逻辑

Prefill 优先。如果 waiting 队列非空，调度器会尝试 admit 尽可能多的新请求：

```python
def schedule(self):
    scheduled = []
    num_seqs = 0
    num_batched_tokens = 0
    
    while self.waiting:
        seq = self.waiting[0]
        # 检查约束
        if num_seqs >= self.max_num_seqs:
            break
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
            break
        if not self.block_manager.can_allocate(seq):
            break
        
        # 准入
        num_seqs += 1
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled.append(seq)
    
    if scheduled:
        return scheduled, True  # True 表示 Prefill
```

准入条件有三个：

| 条件                                                    | 含义                 |
| ------------------------------------------------------- | -------------------- |
| num_seqs < max_num_seqs                                 | 不超过最大请求数     |
| num_batched_tokens + len(seq) <= max_num_batched_tokens | 不超过 token 预算    |
| block_manager.can_allocate(seq)                         | 有足够的 KV Cache 块 |

`num_batched_tokens` 的计算用的是 `len(seq) - seq.num_cached_tokens`，这是因为 Prefix Cache 命中的部分不需要重新计算，只计入实际要处理的 token 数。

### Decode 调度逻辑

如果没有新请求需要 Prefill，就进入 Decode 阶段，处理 running 队列中的请求：

```python
# Decode 阶段
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()
        
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())  # 抢占队尾
            else:
                self.preempt(seq)  # 自抢占
                break
        else:
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled.append(seq)
    
    assert scheduled  # 前进性保证
    self.running.extendleft(reversed(scheduled))
    return scheduled, False  # False 表示 Decode
```

这里有个 `while-else` 结构需要解释：

- `while not can_append(seq)` 循环的意思是：如果当前没有足够的块给这个请求追加新 token，就需要释放一些资源
- 释放资源的方式是抢占：把某个请求踢回 waiting 队列，释放它的块
- 优先抢占 running 队列的尾部（最后加入的请求），如果 running 已经空了，就自抢占
- `else` 分支在 `while` 条件不成立时执行，也就是有足够块的时候，执行实际的 append

### 抢占机制

```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)  # 放到 waiting 队首，下次优先处理
```

抢占会释放这个请求的所有 KV Cache 块，然后把它放回 waiting 队列的最前面。下次 Prefill 时会重新处理这个请求。

这里有个关键的设计决策：被抢占的请求会被放到 waiting 队首而不是队尾。这保证了公平性，不会让一个请求被反复抢占而饿死。

### 前进性保证

注意代码里有一行 `assert scheduled`。这是一个关键的不变量：Decode 阶段必须至少有一个请求被调度。

为什么能保证这一点？因为自抢占机制。最坏情况下，当前请求自己释放自己的块，然后用释放出来的块给自己追加新 token。只要系统里还有请求在 running，就一定能取得进展。

### 后处理

模型执行完之后，需要更新请求状态：

```python
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        
        # 检查终止条件
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

终止条件有两个：遇到 EOS token（除非设置了 ignore_eos），或者达到最大生成长度。

### 调度器的完整流程

把上面的逻辑串起来，调度器的完整工作流程如下：

```python
schedule() 被调用
    ├─ waiting 非空？
    │   ├─ 是：尝试 Prefill admit
    │   │       逐个检查 waiting 队首
    │   │       满足约束则 allocate + RUNNING + 移入 running
    │   │       返回 (scheduled, is_prefill=True)
    │   └─ 否：进入 Decode
    │           逐个处理 running
    │           需要新块但没有？抢占其他请求或自抢占
    │           may_append
    │           返回 (scheduled, is_prefill=False)
    │
postprocess() 被调用
    ├─ 追加 token
    └─ 检查终止，deallocate + FINISHED
```

下一部分我们来实现 BlockManager，解决 KV Cache 的分配问题。

## 页式 KV 缓存

调度器决定了每一步执行哪些请求，但还有一个关键问题没解决：KV Cache 的显存怎么管理？

传统做法是按最大长度预分配。假设最大序列长度是 4096，每个请求一开始就分配 4096 个位置的 KV 存储空间。问题很明显：实际生成长度可能只有几百个 token，大量显存被浪费了。

另一个问题是碎片化。不同请求的长度不同，分配和释放之后，显存会出现很多小块空闲区域，无法被有效利用。

解决方案是分页。把 KV Cache 切成固定大小的块（Block），按需分配，用完就回收。这和操作系统的虚拟内存管理思路一样。

### Block 数据结构

每个 Block 是一个固定大小的存储单元，可以存放 `block_size` 个 token 的 KV 数据：

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0    # 引用计数
        self.hash = -1        # 链式哈希值，-1 表示未定稿
        self.token_ids = []   # 块内的 token（用于校验）

    def update(self, hash, token_ids):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
```

几个设计要点：

1. `ref_count` 支持块共享。多个请求如果有相同的 prefix，可以共享同一个块
2. `hash` 用于 Prefix Cache。满块才计算哈希，未满的块（开放块）hash 为 -1
3. `token_ids` 存储块内 token，用于哈希碰撞时的内容校验

### BlockManager 的基本结构

```python
class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}       # 哈希 → 块ID 映射
        self.free_block_ids = deque(range(num_blocks))  # 空闲块队列
        self.used_block_ids = set()      # 已用块集合
```

`free_block_ids` 和 `used_block_ids` 是互斥的：一个块要么在空闲队列，要么在已用集合。

### 链式哈希

Prefix Cache 的核心是识别相同的 token 序列。朴素做法是直接比较 token 序列，但这样太慢。我们用哈希来加速。

但简单哈希有问题。假设两个序列：

```text
序列 A: [1, 2, 3, 4, 5, 6, 7, 8]  → 块 [1,2,3,4] 和 [5,6,7,8]
序列 B: [1, 2, 3, 4, 9, 10, 11, 12] → 块 [1,2,3,4] 和 [9,10,11,12]
```

如果只对每个块内的 token 做哈希，两个序列的第一个块哈希值一样。但如果：

```text
序列 C: [0, 2, 3, 4, 5, 6, 7, 8]  → 块 [0,2,3,4] 和 [5,6,7,8]
```

序列 C 的第二个块 token 和序列 A 一样，但它们的 prefix 不同，KV 值是不同的（因为 attention 是 causal 的）。

解决方案是链式哈希：计算当前块哈希时，把前一个块的哈希值也加进去。

```python
@classmethod
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

这样，相同的块内容但不同的 prefix，会得到不同的哈希值。只有 prefix 也相同时，哈希值才相同。

### 块分配

当一个新请求需要 Prefill 时，调用 `allocate()`：

```python
def can_allocate(self, seq):
    return len(self.free_block_ids) >= seq.num_blocks

def allocate(self, seq):
    assert not seq.block_table
    h = -1
    cache_miss = False
    
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        # 满块才计算哈希
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)
        
        # 检查是否命中且内容一致
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True
        
        if cache_miss:
            # 未命中，分配新块
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # 命中，复用已有块
            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                block = self._allocate_block(block_id)
        
        # 满块更新哈希映射
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        
        seq.block_table.append(block_id)
```

这段逻辑有点复杂，画个表来理清：

| 情况                            | 处理方式                  | num_cached_tokens |
| ------------------------------- | ------------------------- | ----------------- |
| 哈希命中 + 内容一致 + 块在 used | ref_count++，复用         | += block_size     |
| 哈希命中 + 内容一致 + 块在 free | 分配（从 free 移到 used） | += block_size     |
| 哈希未命中 或 内容不一致        | 分配新块                  | 不变              |

注意 `cache_miss` 是一个”传染”变量：一旦某个块未命中，后续所有块都按未命中处理。这是因为链式哈希的性质，前面不一样，后面即使 token 相同，哈希值也不同。

### 块回收

```python
def deallocate(self, seq):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()

def _allocate_block(self, block_id):
    block = self.blocks[block_id]
    assert block.ref_count == 0
    block.reset()
    self.free_block_ids.remove(block_id)
    self.used_block_ids.add(block_id)
    return block

def _deallocate_block(self, block_id):
    assert self.blocks[block_id].ref_count == 0
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)
```

回收时逆序遍历 block_table，减少引用计数。只有引用计数降到 0 的块才真正回收。这保证了共享块不会被错误释放。

### Decode 阶段的块追加

Decode 阶段每生成一个 token，可能需要分配新块：

```python
def can_append(self, seq):
    # 如果下一个 token 是新块的第一个，需要有空闲块
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq):
    block_table = seq.block_table
    last_block = self.blocks[block_table[-1]]
    
    if len(seq) % self.block_size == 1:
        # 新块的第一个 token，需要分配新块
        assert last_block.hash != -1  # 上一个块必须已定稿
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        block_table.append(block_id)
        
    elif len(seq) % self.block_size == 0:
        # 块刚满，需要定稿（计算哈希并登记）
        assert last_block.hash == -1
        token_ids = seq.block(seq.num_blocks - 1)
        prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
        h = self.compute_hash(token_ids, prefix)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id
        
    else:
        # 还在当前块内，不需要操作
        assert last_block.hash == -1
```

来看一个具体例子，假设 block_size = 4：

| seq 长度 | len % 4 | 动作             |
| -------- | ------- | ---------------- |
| 4        | 0       | 块刚满，定稿块 0 |
| 5        | 1       | 新块首，分配块 1 |
| 6        | 2       | 无操作           |
| 7        | 3       | 无操作           |
| 8        | 0       | 块刚满，定稿块 1 |
| 9        | 1       | 新块首，分配块 2 |

### 开放块与定稿块

这里引入两个概念：

**开放块**：hash = -1，还在被写入，内容可能变化 **定稿块**：hash != -1，内容固定，可以被其他请求共享

为什么要区分？因为 Prefix Cache 的正确性依赖于此。只有定稿块的内容是确定的，才能安全地被其他请求复用。如果一个块还在被写入，另一个请求来复用它，后续写入会破坏共享块的内容。

块的状态转换：

```text
分配时：开放块（hash = -1）
     ↓
填满时：定稿块（hash = 计算值，登记到映射）
     ↓
回收时：ref_count--，降到 0 则移回 free
```

### 数值化的例子

假设 block_size = 256，两个请求 S1 和 S2：

```text
S1: 600 个 token，前 512 和 S2 相同
S2: 520 个 token，前 512 和 S1 相同
```

S1 先 Prefill：

```text
块划分: [0..255], [256..511], [512..599]
分配:
  块 0 (256 tokens) → 计算 h0，分配 block_id=a，登记 hash_to_block[h0]=a
  块 1 (256 tokens) → 计算 h1(prefix=h0)，分配 block_id=b，登记 hash_to_block[h1]=b  
  块 2 (88 tokens)  → 开放块，hash=-1，分配 block_id=c
S1.block_table = [a, b, c]
S1.num_cached_tokens = 0
```

S2 后 Prefill：

```text
块划分: [0..255], [256..511], [512..519]
分配:
  块 0 (256 tokens) → 计算 h0，查到 block_id=a，内容一致 → 命中！
                      ref_count++ (a.ref_count = 2)
                      num_cached_tokens += 256
  块 1 (256 tokens) → 计算 h1，查到 block_id=b，内容一致 → 命中！
                      ref_count++ (b.ref_count = 2)
                      num_cached_tokens += 256
  块 2 (8 tokens)   → 开放块，不计算哈希，分配新块 block_id=d
S2.block_table = [a, b, d]
S2.num_cached_tokens = 512
```

S2 命中了 S1 的前两个块，省去了 512 个 token 的 KV 计算。

### 物理存储布局

实际的 KV Cache 是一个大张量：

```python
kv_cache = torch.empty(
    2,                          # K 和 V
    num_layers,                 # 层数
    num_blocks,                 # 总块数
    block_size,                 # 每块 token 数
    num_kv_heads // tp_size,    # KV head 数（考虑张量并行）
    head_dim                    # head 维度
)
```

每层的 Attention 模块持有这个大张量对应层的切片：

```python
module.k_cache = kv_cache[0, layer_id]  # shape: [num_blocks, block_size, num_kv_heads, head_dim]
module.v_cache = kv_cache[1, layer_id]
```

block_table 存的是逻辑块到物理块的映射。比如 `seq.block_table = [5, 12, 3]` 表示：

- 序列的第 0 个逻辑块存在物理块 5
- 序列的第 1 个逻辑块存在物理块 12
- 序列的第 2 个逻辑块存在物理块 3

### 显存预算计算

系统启动时，需要根据可用显存计算能分配多少块：

```python
def allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    
    # 每块占用的字节数
    block_bytes = (2 * num_layers * block_size * 
                   num_kv_heads // tp_size * 
                   head_dim * dtype.itemsize)
    
    # 可用显存 = 总量 × 利用率 - 已用 - (峰值 - 当前)
    available = total * gpu_memory_utilization - used - peak + current
    num_blocks = int(available) // block_bytes
    assert num_blocks > 0
```

这个计算考虑了模型权重、激活值峰值等其他显存占用，只把剩余部分分给 KV Cache。

下一部分我们来看 ModelRunner，把调度器选出的请求送入模型执行。

## 模型执行后端

调度器选出了本步要执行的请求，BlockManager 分配了 KV Cache 的物理块。现在需要把这些信息转换成模型能接受的输入格式，执行前向传播，然后采样得到下一个 token。

这就是 ModelRunner 的职责。

### 执行上下文

Prefill 和 Decode 两个阶段需要的元信息不同。我们用一个 Context 数据结构来统一管理：

```python
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor = None   # Query 累计长度
    cu_seqlens_k: torch.Tensor = None   # Key 累计长度  
    max_seqlen_q: int = 0               # 最长 Query 长度
    max_seqlen_k: int = 0               # 最长 Key 长度
    slot_mapping: torch.Tensor = None   # KV 写入的物理槽位
    context_lens: torch.Tensor = None   # 每个序列的上下文长度（Decode 用）
    block_tables: torch.Tensor = None   # 物理块表
```

Context 是一个全局单例，通过 `set_context` / `get_context` / `reset_context` 来操作：

```python
_CONTEXT = Context()

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, 
                max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, 
                context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, 
                       max_seqlen_q, max_seqlen_k, slot_mapping,
                       context_lens, block_tables)

def get_context():
    return _CONTEXT

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
```

为什么用全局变量？因为 Context 需要被模型的各个层访问（主要是 Attention 层），而模型的 forward 签名是固定的 `(input_ids, positions)`。用全局变量可以在不改变模型接口的情况下传递额外信息。

### Prefill 输入准备

Prefill 阶段要处理多个变长序列。Flash Attention 的 varlen 接口需要把所有序列的 token 拼成一个长向量，然后用累计长度数组来区分边界。

```python
def prepare_prefill(self, seqs):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    block_tables = None
    
    for seq in seqs:
        seqlen = len(seq)
        # 只处理未缓存的部分
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(list(range(seq.num_cached_tokens, seqlen)))
        
        seqlen_q = seqlen - seq.num_cached_tokens  # Query 长度（未缓存部分）
        seqlen_k = seqlen                          # Key 长度（完整序列）
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)
        
        # 构造 slot_mapping：只包含要写入的槽位
        if not seq.block_table:  # warmup 时没有 block_table
            continue
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            start = seq.block_table[i] * self.block_size
            if i != seq.num_blocks - 1:
                end = start + self.block_size
            else:
                end = start + seq.last_block_num_tokens
            slot_mapping.extend(list(range(start, end)))
    
    # Prefix Cache 判断
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
        block_tables = self.prepare_block_tables(seqs)
    
    # 转换为 GPU 张量
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    
    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
                slot_mapping, None, block_tables)
    return input_ids, positions
```

这里有几个关键点：

**cu_seqlens_q 和 cu_seqlens_k 的区别**

当 Prefix Cache 命中时，序列的一部分 token 已经有 KV Cache 了，不需要重新计算。所以 Query 只包含未缓存的部分，而 Key 包含完整序列（缓存的部分从 Cache 读取）。

举个例子，三个序列的长度分别是 100、200、150，其中第一个序列命中了 50 个 token 的缓存：

```python
cu_seqlens_q = [0, 50, 250, 400]   # Query: 50 + 200 + 150 = 400
cu_seqlens_k = [0, 100, 300, 450]  # Key:  100 + 200 + 150 = 450
```

**slot_mapping 的语义**

slot_mapping 是一个一维数组，存储的是本步要写入 KV Cache 的物理槽位索引。

物理槽位 = block_id × block_size + offset_in_block

例如 block_size = 256，序列的 block_table = [5, 12]，num_cached_tokens = 256（第一个块已缓存），总长度 = 300：

- 需要写入的是 token 256-299，共 44 个
- 它们在第二个块（block_id = 12）的位置 0-43
- slot_mapping = [12×256+0, 12×256+1, …, 12×256+43] = [3072, 3073, …, 3115]

**Prefix Cache 启用条件**

`cu_seqlens_k[-1] > cu_seqlens_q[-1]` 表示总 Key 长度大于总 Query 长度，说明有缓存命中。这时需要构造 block_tables，让 Attention 知道去哪里读取缓存的 KV。

### Decode 输入准备

Decode 阶段每个序列只有一个 token 输入：

```python
def prepare_decode(self, seqs):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []
    
    for seq in seqs:
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        # 最后一个槽位
        slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
    
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    block_tables = self.prepare_block_tables(seqs)
    
    set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
    return input_ids, positions
```

Decode 的 slot_mapping 每个序列只有一个元素，就是新 token 的 KV 要写入的位置。

### block_tables 准备

block_tables 是一个二维张量，每行是一个序列的物理块列表，需要 pad 到相同长度：

```python
def prepare_block_tables(self, seqs):
    max_len = max(len(seq.block_table) for seq in seqs)
    block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
    block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    return block_tables
```

### 模型执行

准备好输入之后，调用模型前向传播：

```python
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # CUDA Graph 路径
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        
        # 写入 staging 张量
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

执行路径的选择逻辑：

| 条件          | 路径       | 原因                                                      |
| ------------- | ---------- | --------------------------------------------------------- |
| is_prefill    | Eager      | Prefill 的 shape 变化大，不适合 Graph                     |
| enforce_eager | Eager      | 用户强制指定                                              |
| bs > 512      | Eager      | batch 太大，Graph 收益不明显                              |
| 其他          | CUDA Graph | Decode 的 shape 固定，Graph 能显著降低 kernel launch 开销 |

### CUDA Graph 捕获

CUDA Graph 的原理是把一系列 CUDA kernel 调用”录制”下来，之后可以一次性重放，省去每次的 launch 开销。对于 Decode 阶段这种计算量小但调用频繁的场景特别有效。

```python
@torch.inference_mode()
def capture_cudagraph(self):
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
    
    # 预分配 staging 张量
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, self.config.hf_config.hidden_size)
    
    # 要捕获的 batch size 列表
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None
    
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=slot_mapping[:bs], 
                    context_lens=context_lens[:bs], block_tables=block_tables[:bs])
        
        # warmup
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        
        # capture
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()
    
    self.graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
    )
```

几个要注意的点：

1. **batch size 列表**：`[1, 2, 4, 8, 16, 32, ...]`。运行时选择第一个 >= 实际 bs 的图。比如实际 bs=5 会用 bs=8 的图，浪费一点计算但避免了太多图。
2. **逆序捕获**：从大到小捕获，这样大图先分配显存，小图可以复用。`graph_pool` 让多个图共享同一块显存。
3. **staging 张量**：Graph 要求输入输出张量的地址不变。所以预分配固定大小的张量，每次把实际数据拷贝进去。
4. **warmup**：捕获前先跑一次，让 PyTorch 完成编译和显存分配。

### 完整的 run 流程

```python
def run(self, seqs, is_prefill):
    # 准备输入
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    
    # 执行模型
    logits = self.run_model(input_ids, positions, is_prefill)
    
    # 采样（只在 rank 0）
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    
    reset_context()
    return token_ids
```

### ModelRunner 初始化

```python
class ModelRunner:
    def __init__(self, config, rank, event):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        
        # NCCL 初始化
        dist.init_process_group("nccl", "tcp://localhost:2333", 
                                world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        
        # 设置默认 dtype 和 device
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # 构建模型
        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        
        # 预热和资源分配
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        # 还原默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        
        # 多卡 IPC 设置
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()  # 非 rank0 进入等待循环
```

初始化顺序：

```text
NCCL init → 构建模型 → 加载权重 → warmup → 分配 KV Cache → 捕获 Graph → IPC 设置
```

warmup 的作用是稳定显存峰值。用”最坏情况”的输入跑一次，让 PyTorch 完成所有的内存分配和编译：

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batched_tokens = self.config.max_num_batched_tokens
    max_model_len = self.config.max_model_len
    num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
    self.run(seqs, True)
    torch.cuda.empty_cache()
```

### 张量形状总结

| 阶段    | 张量         | 形状                     | dtype |
| ------- | ------------ | ------------------------ | ----- |
| Prefill | input_ids    | [ΣQ]                     | int64 |
| Prefill | positions    | [ΣQ]                     | int64 |
| Prefill | cu_seqlens_q | [bs+1]                   | int32 |
| Prefill | cu_seqlens_k | [bs+1]                   | int32 |
| Prefill | slot_mapping | [Σ写入槽位]              | int32 |
| Prefill | block_tables | [bs, max_blocks] 或 None | int32 |
| Decode  | input_ids    | [bs]                     | int64 |
| Decode  | positions    | [bs]                     | int64 |
| Decode  | slot_mapping | [bs]                     | int32 |
| Decode  | context_lens | [bs]                     | int32 |
| Decode  | block_tables | [bs, max_blocks]         | int32 |

下一部分我们进入 Attention 层，看看 Flash Attention 如何与分页 KV Cache 配合工作。

## 注意力机制的工程实现

Attention 层是推理框架的核心。它需要完成两件事：把当前计算的 KV 写入缓存，然后执行注意力计算。难点在于如何与分页的 KV Cache 高效配合。

### Attention 模块的结构

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q, k, v):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # 写入 KV Cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        # 执行 Attention
        if context.is_prefill:
            if context.block_tables is not None:  # Prefix Cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale, causal=True, block_table=context.block_tables
            )
        else:  # Decode
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens, block_table=context.block_tables,
                softmax_scale=self.scale, causal=True
            )
        return o
```

### KV Cache 写入

slot_mapping 告诉我们每个 token 的 KV 应该写到物理缓存的哪个位置。这个写入操作用 Triton 实现：

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

这个内核每个 program 处理一个 token 的 KV 写入。`D = num_kv_heads * head_dim` 是每个 token 的 KV 向量长度。

调用时需要做一些 stride 检查：

```python
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    
    # 确保内存布局正确
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), 
                                k_cache, v_cache, slot_mapping, D)
```

为什么 slot = -1 时直接返回？这是为了处理 CUDA Graph 的 padding。Graph 捕获时用的是固定大小的张量，实际 batch 可能更小，多余的位置 slot_mapping 填 -1，内核跳过这些位置。

### Flash Attention 接口

**Prefill：flash_attn_varlen_func**

处理变长序列的批次。所有序列的 Q/K/V 拼成一个长向量，用 cu_seqlens 标记边界：

```python
o = flash_attn_varlen_func(
    q,                              # [total_q, num_heads, head_dim]
    k,                              # [total_k, num_kv_heads, head_dim]
    v,                              # [total_k, num_kv_heads, head_dim]
    cu_seqlens_q=cu_seqlens_q,      # [batch_size + 1]
    cu_seqlens_k=cu_seqlens_k,      # [batch_size + 1]
    max_seqlen_q=max_seqlen_q,      # int
    max_seqlen_k=max_seqlen_k,      # int
    softmax_scale=scale,
    causal=True,
    block_table=block_tables        # [batch_size, max_blocks] 或 None
)
```

当 block_table 不为 None 时，Flash Attention 会从分页的 KV Cache 中读取 K/V，而不是用传入的 k/v 参数。这就是 Prefix Cache 的实现方式。

**Decode：flash_attn_with_kvcache**

每个序列只有一个 query，但要和完整的历史 KV 做 attention：

```python
o = flash_attn_with_kvcache(
    q.unsqueeze(1),                 # [batch_size, 1, num_heads, head_dim]
    k_cache,                        # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache,                        # [num_blocks, block_size, num_kv_heads, head_dim]
    cache_seqlens=context_lens,     # [batch_size]
    block_table=block_tables,       # [batch_size, max_blocks]
    softmax_scale=scale,
    causal=True
)
```

注意 q 需要 unsqueeze(1) 增加一个 seqlen 维度。cache_seqlens 告诉 Flash Attention 每个序列当前的长度，避免读取 padding 位置的无效数据。

### Prefix Cache 的 Attention 路径

Prefill 阶段，如果有 Prefix Cache 命中，流程如下：

```python
1. prepare_prefill 检测到 cu_seqlens_k[-1] > cu_seqlens_q[-1]
   → 说明部分 token 已缓存
   → 构造 block_tables

2. Attention.forward:
   a. store_kvcache 写入未缓存部分的 KV（slot_mapping 只包含未缓存的槽位）
   b. block_tables 不为 None，所以把 k, v 替换为 k_cache, v_cache
   c. flash_attn_varlen_func 通过 block_table 读取分页的完整 KV
```

来看一个具体例子。序列长度 300，前 256 个 token 命中缓存（一个完整块）：

```text
cu_seqlens_q = [0, 44]     # Query 只有 44 个 token（300 - 256）
cu_seqlens_k = [0, 300]    # Key 有 300 个 token
slot_mapping = [3072, 3073, ..., 3115]  # 44 个槽位，块 12 的位置 0-43
block_tables = [[5, 12]]   # 两个块：块 5（已缓存），块 12（新写入）
```

Attention 执行时：

- Q 来自当前计算的 44 个 token
- K/V 来自缓存，通过 block_table 索引读取全部 300 个 token 的 KV

### KV Cache 的内存布局

每层的 KV Cache 形状是：

```python
k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
```

物理槽位的计算：

```python
slot = block_id * block_size + offset_in_block
```

在缓存中的位置：

```python
k_cache[block_id, offset_in_block, :, :]  # 一个 token 的所有 KV head
```

Triton 内核按扁平化的方式写入：

```python
cache_offsets = slot * D + tl.arange(0, D)
```

这里 `D = num_kv_heads * head_dim`，相当于把后两个维度展平了。

### Q/K/V 的 reshape

在 Attention 之前，QKV 投影的输出需要 reshape 成正确的形状：

```python
# 在 Qwen3Attention.forward 中
qkv = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
q = q.view(-1, self.num_heads, self.head_dim)
k = k.view(-1, self.num_kv_heads, self.head_dim)
v = v.view(-1, self.num_kv_heads, self.head_dim)
```

这里 `self.q_size = num_heads * head_dim`，`self.kv_size = num_kv_heads * head_dim`。

GQA（Grouped Query Attention）模型中 `num_kv_heads < num_heads`，多个 query head 共享同一组 KV head。Flash Attention 内部处理了这个广播。

### 缩放因子

标准的 attention 缩放是 `1 / sqrt(head_dim)`：

```python
self.scaling = self.head_dim ** -0.5
```

这个值在初始化时计算好，传给 `flash_attn_*_func` 的 `softmax_scale` 参数。

### RoPE 的位置

RoPE（旋转位置编码）在 Attention 前对 Q 和 K 应用：

```python
# 在 Qwen3Attention.forward 中
q, k = self.rotary_emb(positions, q, k)
o = self.attn(q, k, v)
```

注意 RoPE 是在写入 KV Cache 之前应用的。所以缓存中存的是已经编码过位置信息的 K。这意味着同一个 token 在不同位置的 K 是不同的。

但这不影响 Prefix Cache 的正确性。因为 Prefix Cache 的哈希是基于 token 序列计算的，相同的 token 序列一定对应相同的位置，所以 K 值也一定相同。

下一部分我们来实现张量并行，让模型能在多卡上运行。

## 张量并行

当模型太大单卡放不下，或者想提升吞吐量时，需要把模型切分到多张卡上。张量并行（Tensor Parallelism）是最常用的方案，核心思想是把矩阵乘法拆分到多卡并行计算。

### 列并行与行并行

线性层的计算是 `Y = XW + b`，其中 X 是输入，W 是权重，b 是偏置。

**列并行（Column Parallel）**：把权重 W 按列切分

```python
W = [W_0 | W_1 | ... | W_{n-1}]   每个 rank 持有 W_i

每个 rank 计算: Y_i = X @ W_i
最终输出: Y = [Y_0 | Y_1 | ... | Y_{n-1}]  (沿特征维拼接)
```

输入 X 在所有 rank 上是完整的（或者说每个 rank 都有一份拷贝），输出 Y 是分片的。

**行并行（Row Parallel）**：把权重 W 按行切分

```python
W = [W_0]
    [W_1]
    [...]
    [W_{n-1}]   每个 rank 持有 W_i

输入也要对应切分: X = [X_0 | X_1 | ... | X_{n-1}]
每个 rank 计算: Y_i = X_i @ W_i
最终输出: Y = sum(Y_0, Y_1, ..., Y_{n-1})  (需要 all_reduce)
```

行并行需要通信来聚合结果。

### 为什么需要两种并行

Transformer 的 FFN 层有两个线性变换：

```text
hidden → intermediate → hidden
```

如果都用列并行，第一层的输出是分片的，第二层的输入也是分片的，需要先 all_gather 再计算。如果都用行并行，每层都要 all_reduce。

最优方案是交替使用：第一层用列并行，输出分片；第二层用行并行，直接接收分片输入，最后 all_reduce 一次。这样每个 FFN block 只需要一次通信。

### LinearBase 基类

```python
def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    def __init__(self, input_size, output_size, bias=False, tp_dim=None):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)
```

`tp_dim` 指定切分维度。`weight_loader` 是一个自定义的加载函数，用于从完整权重中提取当前 rank 的分片。

### ColumnParallelLinear

```python
class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)

    def weight_loader(self, param, loaded_weight):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```

构造时 output_size 除以 tp_size，每个 rank 只持有一部分输出维度。`weight_loader` 从完整权重中切出对应的分片。

前向传播就是普通的 `F.linear`，不需要通信。

### RowParallelLinear

```python
class RowParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)

    def weight_loader(self, param, loaded_weight):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

构造时 input_size 除以 tp_size。注意权重的切分维度是 1（输入维度），因为 PyTorch 的线性层权重形状是 `[output_size, input_size]`。

前向传播后需要 `all_reduce` 聚合各 rank 的部分结果。bias 只在 rank 0 加，避免重复加。

### MergedColumnParallelLinear

MLP 层的 gate 和 up 投影可以合并成一个大矩阵：

```python
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size, output_sizes, bias=False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
```

`output_sizes` 是一个列表，比如 `[intermediate_size, intermediate_size]` 表示 gate 和 up 两个投影的输出维度。

`weight_loader` 多了一个 `loaded_shard_id` 参数，指定当前加载的是第几个子模块的权重（0 表示 gate，1 表示 up）。

### QKVParallelLinear

Attention 的 QKV 投影更复杂，因为 Q 和 KV 的 head 数可能不同（GQA）：

```python
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads=None, bias=False):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # v
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
```

QKV 合并后的布局是 `[Q | K | V]`，每个部分按 head 数切分。`loaded_shard_id` 用字符串 “q”/“k”/“v” 来区分。

### VocabParallelEmbedding

Embedding 层的词表维度也可以切分：

```python
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param, loaded_weight):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x):
        if self.tp_size > 1:
            # 只处理属于当前 rank 的词表范围
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 不在范围内的位置置零，然后 all_reduce 聚合
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y
```

每个 rank 只持有词表的一部分。前向时：

1. 检查输入 token id 是否在当前 rank 的范围内
2. 在范围内的正常查表，不在范围内的置零
3. all_reduce 聚合所有 rank 的结果

### ParallelLMHead

LMHead 和 Embedding 类似，但有两个特殊处理：

```python
class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings, embedding_dim, bias=False):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x):
        context = get_context()
        if context.is_prefill:
            # Prefill 时只取每个序列的最后一个位置
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        
        logits = F.linear(x, self.weight)
        
        if self.tp_size > 1:
            # rank 0 gather 所有分片，拼接成完整词表
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, dst=0)
            logits = torch.cat(all_logits, dim=-1) if self.tp_rank == 0 else None
        
        return logits
```

Prefill 时，hidden states 包含所有序列的所有 token，但我们只需要每个序列最后一个位置的 logits（用于采样下一个 token）。`cu_seqlens_q[1:] - 1` 正好是每个序列的最后一个索引。

多卡时，每个 rank 计算的 logits 只覆盖部分词表。用 `dist.gather` 把所有分片收集到 rank 0，拼接成完整的词表维度。

### 通信模式总结

| 层类型                 | 切分方式 | 通信操作           | 时机       |
| ---------------------- | -------- | ------------------ | ---------- |
| ColumnParallel         | 输出维   | 无                 | -          |
| RowParallel            | 输入维   | all_reduce         | forward 后 |
| VocabParallelEmbedding | 词表维   | all_reduce         | forward 后 |
| ParallelLMHead         | 词表维   | gather (到 rank 0) | forward 后 |

### 多卡 IPC 通信

除了 NCCL 的集合通信，还需要 rank 0 向其他 rank 广播控制信息（比如要执行哪个方法、参数是什么）。这里用共享内存实现：

```python
def write_shm(self, method_name, *args):
    assert self.world_size > 1 and self.rank == 0
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()

def read_shm(self):
    assert self.world_size > 1 and self.rank > 0
    self.event.wait()
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args
```

协议很简单：

- 前 4 字节存数据长度
- 后续存 pickle 序列化的 `[method_name, *args]`
- 用 Event 做同步

非 rank 0 的进程在初始化后进入一个循环，等待 rank 0 的指令：

```python
def loop(self):
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)
        if method_name == "exit":
            break
```

### 权重加载

权重加载时需要根据 packed_modules_mapping 处理合并权重：

```python
def load_model(model, path):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 检查是否是需要拆分的合并权重
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 普通权重
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

模型定义中的 `packed_modules_mapping`：

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

HuggingFace 的权重是分开的（q_proj、k_proj、v_proj），但我们的模型把它们合并成了 qkv_proj。加载时需要把分开的权重分别写入合并模块的对应位置。

下一部分我们来看完整的模型层实现，把这些并行层组装成 Transformer。

## 模型层实现

有了并行层、Attention、ModelRunner，我们现在可以像搭积木一样，把 Qwen3 模型组装起来。

### 模型结构概览

```text
Qwen3ForCausalLM
  └── Qwen3Model
        ├── VocabParallelEmbedding
        ├── LayerList
        │     └── Qwen3DecoderLayer
        │           ├── RMSNorm (input)
        │           ├── Qwen3Attention
        │           ├── RMSNorm (post_attn)
        │           └── Qwen3MLP
        └── RMSNorm (final)
  └── ParallelLMHead
```

### Decoder Layer 实现

每个 Decoder Layer 包含两个残差块：Attention 和 MLP。

```python
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual=None):
        # Block 1: Attention
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states)
        
        # Block 2: MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual
```

这里有一个优化：**Residual 融合**。

标准的 LayerNorm 是 `y = norm(x + res)`。在 Qwen3 的实现中，LayerNorm 层接收 `(x, res)`，返回 `(normed_x, updated_res)`。这利用了 PyTorch 的 `add_` 原地操作来减少显存开销和 Kernel 启动。

### RMSNorm 实现

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def add_rms_forward(self, x, residual):
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())  # 累加残差
        residual = x.to(orig_dtype)           # 更新残差供下一层用
        
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual
```

用了 `torch.compile` 装饰器。RMSNorm 是内存带宽受限的操作（Element-wise），编译后可以把加法、平方、均值、开方、乘法等操作融合成一个 Kernel，显著提升性能。

### MLP 实现

Qwen3 的 MLP 使用 SwiGLU 激活函数，结构是 `(gate * silu(up)) @ down`。

```python
class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        # gate 和 up 合并成一个列并行层
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False
        )
        # down 是行并行层
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
```

`SiluAndMul` 也是一个被编译的算子：

```text
@torch.compile
def forward(self, x):
    x, y = x.chunk(2, -1)  # 切分成 gate 和 up
    return F.silu(x) * y
```

### Attention 模块实现

```python
class Qwen3Attention(nn.Module):
    def __init__(self, ...):
        # ... 初始化逻辑 ...
        self.qkv_proj = QKVParallelLinear(...)
        self.o_proj = RowParallelLinear(...)
        self.rotary_emb = get_rope(...)
        self.attn = Attention(...)
        
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        if not self.qkv_bias:  # Qwen 特有的 QK Norm
            q = self.q_norm(q)
            k = self.k_norm(k)
            
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output
```

### 主模型实现

```python
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = { ... }  # 权重映射表

    def __init__(self, config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        # 权重绑定
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

注意 forward 只返回 hidden_states，不返回 logits。`compute_logits` 是单独的方法。这是为了适应 ModelRunner 的两阶段设计：

- Prefill/Eager 模式：调用 `compute_logits(model(input_ids))`
- Graph 模式：先重放 graph 得到 hidden_states，再调用 `compute_logits`

为什么不把 LMHead 放入 Graph？

1. LMHead 的输出太大（Batch × VocabSize），显存占用高
2. Prefill 时只需要最后位置的 logits，这个 gather 操作比较复杂
3. Decode 时 Batch 小，LMHead 不算瓶颈

### 权重绑定

很多模型（如 Qwen, Llama）的 Embedding 层和 LMHead 层共享权重。初始化时需要手动把数据指针指过去：

```python
self.lm_head.weight.data = self.model.embed_tokens.weight.data
```

这样两个模块共享同一块物理显存。

### 编译与性能

我们在关键的瓶颈位置用了 `torch.compile`：

- RMSNorm：Element-wise 算子融合
- SiluAndMul：激活函数融合
- RotaryEmbedding：复杂的索引和三角函数计算
- Sampler：Softmax + Argmax 融合

这些也是 PyTorch 2.0 推荐的优化点。对于矩阵乘法（Linear 层），cuBLAS 已经优化得很好，编译收益不大。

下一部分我们来看最后一块拼图：采样器和主循环验证。

## 采样与终止

模型前向传播输出 logits 之后，需要采样得到下一个 token。这一步看起来简单，但有几个细节值得展开。

### 温度采样

最常用的采样方式是温度采样：

```python
logits = logits / temperature
probs = softmax(logits)
next_token = categorical(probs)
```

温度越高，分布越平坦，生成越随机；温度越低，分布越尖锐，越倾向于选概率最高的 token。

问题是 `categorical` 采样需要生成随机数，然后二分查找累积分布函数，计算开销不小。有没有更高效的方式？

### Gumbel-Max 技巧

一个等价但更高效的实现：

```python
@torch.compile
def forward(self, logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    return sample_tokens
```

这利用了一个数学性质：如果 `G_i ~ Gumbel(0, 1)`，那么 `argmax(log(p_i) + G_i)` 等价于从 `Categorical(p)` 采样。

而 `Gumbel(0, 1)` 分布可以通过 `-log(Exp(1))` 生成。进一步简化，`log(p_i) - log(Exp_i)` 等价于 `log(p_i / Exp_i)`，取 argmax 等价于对 `p_i / Exp_i` 取 argmax。

所以最终的实现是：

1. Softmax 得到概率
2. 除以 Exponential(1) 噪声
3. 取 argmax

这个操作是纯张量运算，可以被 `torch.compile` 融合成一个高效的 Kernel。

### 温度下界

注意采样参数的约束：

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

为什么禁止 temperature = 0（贪婪采样）？

1. 除以零会产生 NaN
2. 即使特殊处理，贪婪采样的计算图和温度采样不同，会导致 `torch.compile` 生成不同的 Kernel，影响 Graph 捕获

如果需要贪婪采样，可以设一个很小的温度（如 0.01），效果基本等价。

### 终止条件

在调度器的 postprocess 中检查终止：

```python
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        
        # 终止条件
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

两个终止条件：

| 条件                                | 含义             |
| ----------------------------------- | ---------------- |
| token_id == eos and not ignore_eos  | 生成了结束符     |
| num_completion_tokens == max_tokens | 达到最大生成长度 |

`ignore_eos` 用于需要强制生成固定长度的场景（比如 Benchmark）。

------

## 串联与验证

现在把所有组件串起来，看完整的推理流程。

### LLMEngine 主循环

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        config = Config(model, **kwargs)
        
        # 启动多进程（如果 TP > 1）
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # rank 0 的 ModelRunner
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def generate(self, prompts, sampling_params, use_tqdm=True):
        # 添加请求
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        # 主循环
        outputs = {}
        while not self.scheduler.is_finished():
            output, num_tokens = self.step()
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
        
        # 按 seq_id 排序输出
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} 
                   for token_ids in outputs]
        return outputs
```

### 一步执行的完整流程

```text
step() 被调用
    │
    ├─ scheduler.schedule()
    │   ├─ waiting 非空？尝试 Prefill admit
    │   └─ 否则 Decode，必要时抢占
    │
    ├─ model_runner.call("run", seqs, is_prefill)
    │   ├─ prepare_prefill() 或 prepare_decode()
    │   │   └─ 构造 input_ids, positions, slot_mapping, ...
    │   │   └─ set_context(...)
    │   │
    │   ├─ run_model()
    │   │   ├─ Prefill/Eager: model(input_ids) → compute_logits()
    │   │   └─ Graph: staging → replay → compute_logits()
    │   │
    │   ├─ sampler(logits, temperatures)
    │   │   └─ 只在 rank 0 执行
    │   │
    │   └─ reset_context()
    │
    └─ scheduler.postprocess(seqs, token_ids)
        ├─ 追加 token
        ├─ 检查终止条件
        └─ 终止的序列 deallocate + FINISHED
```

### 关键配置参数

```python
@dataclass
class Config:
    model: str                          # 模型路径
    max_num_batched_tokens: int = 16384 # 单步最大 token 数
    max_num_seqs: int = 512             # 最大并发请求数
    max_model_len: int = 4096           # 最大序列长度
    gpu_memory_utilization: float = 0.9 # 显存利用率
    tensor_parallel_size: int = 1       # 张量并行度
    enforce_eager: bool = False         # 强制 Eager 模式
    kvcache_block_size: int = 256       # KV Cache 块大小
```

参数影响分析：

| 参数                   | 影响                                  |
| ---------------------- | ------------------------------------- |
| max_num_batched_tokens | Prefill 吞吐上限，太大可能 OOM        |
| max_num_seqs           | 并发度上限，影响 Decode 吞吐          |
| max_model_len          | 决定 Graph 捕获时的 block_tables 大小 |
| gpu_memory_utilization | KV Cache 块数，太高容易 OOM           |
| kvcache_block_size     | 块粒度，影响缓存命中率和元数据开销    |
| enforce_eager          | 调试用，关闭 Graph 便于排查问题       |

### 性能测试方法

```python
def benchmark():
    num_seqs = 256
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, 1024))] 
                        for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, 
                                       max_tokens=randint(100, 1024)) 
                       for _ in range(num_seqs)]
    
    # Warmup
    llm.generate(["Benchmark: "], SamplingParams())
    
    # 计时
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
```

------

## 总结

我们从一个朴素的推理循环出发，逐步引入了：

1. **两阶段执行**：区分 Prefill（计算密集）和 Decode（访存密集）
2. **动态调度**：Continuous Batching，抢占机制保证前进性
3. **页式 KV Cache**：按需分配，链式哈希实现 Prefix Cache
4. **CUDA Graph**：降低 Decode 阶段的 Kernel Launch 开销
5. **张量并行**：多卡推理，列并行 + 行并行组合
6. **算子优化**：torch.compile 融合，Triton 自定义 Kernel

整个框架的代码量约 1200 行，但覆盖了高性能 LLM 推理的核心技术。性能上可以达到与成熟框架相当的水平。

这套架构的设计思想是通用的，可以扩展到其他模型（调整 packed_modules_mapping 和模型定义），可以增加更多采样策略（修改 Sampler），可以实现更复杂的调度算法（修改 Scheduler）。