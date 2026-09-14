# infra-learning

从文件存储出发，理解 AI Infra。

[![GitHub stars](https://img.shields.io/github/stars/SnowLegend-star/infra-learning?style=flat-square)](https://github.com/SnowLegend-star/infra-learning)

这是我的 AI Infra 学习记录。它不是一套已经定稿的教材，而是我从文件存储出发，逐步走向大模型训练、推理和分布式系统的阅读笔记、论文译文、源码分析与工程案例。

这个项目的主线是：**数据如何被组织、搬运、缓存、调度和可靠地服务于计算**。文件存储中的很多问题，会在 AI Infra 中以新的形式重新出现：文件系统中的数据面和控制面，对应推理系统中的数据搬运和请求调度；存储系统中的块管理、元数据和缓存，对应 LLM 服务中的 KV Cache、PagedAttention 和 Prefix Cache；RDMA 和高性能 I/O，则连接了 GPU、主机内存、网络与分布式存储。

## 从哪里开始

不需要一次读完所有文章。建议先根据自己的背景选择一条路线：

- **没有存储或 AI Infra 背景**：先读[漫谈文件存储](<存储领域高手/漫谈文件存储.md>)，再读 [Linux IO 模型详解](<存储领域高手/Linux IO模型详解.md>)、[传统文件传输发展历程](<存储领域高手/传统文件传输发展历程.md>) 和 [FUSE 模块学习](<存储领域高手/Fuse模块学习.md>)。
- **已经熟悉分布式存储**：重点阅读 [TafDB 设计思路](<存储领域高手/TafDB设计思路.md>)、[CFS 论文背后的故事](<存储领域高手/如何将千亿文件放进一个文件系统，EuroSys'23 CFS 论文背后的故事.md>)、[Mantle 译文](<存储领域高手/Mantle译文md.md>) 和 [ByteFUSE 的演进与落地](<存储领域高手/ByteFUSE分布式文件系统的演进与落地.md>)。
- **想进入 LLM 推理系统**：先读 [Transformer 深入浅出介绍](<AI infra/Transformer深入浅出介绍.md>)、[Attention 从 MHA 到 DeepSeek MLA](<AI infra/一文了解Attention，从MHA到DeepSeek MLA.md>)，然后读 [PagedAttention 译文](<AI infra/PagedAttention译文.md>)、[vLLM 源码解析](<AI infra/图解大模型计算加速系列：vLLM源码解析1，2.md>) 和 [nano-vLLM 分析](<AI infra/nano—vLLM分析.md>)。
- **关注分布式推理和 KV Cache**：沿着 [PD 分离概述](<AI infra/PD(Prefill&Decode)分离概述.md>) → [Mooncake 译文](<AI infra/Mooncake译文.md>) → [RDMA 技术架构](<AI infra/RDMA技术架构深度解析.md>) 的顺序阅读，再结合 [NVIDIA Dynamo](<AI infra/NVIDIA Dynamo 分布式AI推理的高效引擎.md>) 和存储目录中的 [HiCache 官方文档](<存储领域高手/Hicache官方文档.md>)。

如果只想先建立全局认识，可以按下面的顺序浏览：

```text
文件存储基础
    -> 分布式文件系统与元数据
    -> Linux I/O、FUSE 与 RDMA
    -> Transformer 与 GPU 计算
    -> KV Cache、PagedAttention 与 vLLM
    -> PD 分离、分布式 KV Cache 与高性能网络
```

## 项目地图

```text
infra-learning/
├── 存储领域高手/    文件系统、I/O、元数据、共识、RDMA 与生产案例
└── AI infra/        Transformer、CUDA、训练并行、LLM 推理与 KV Cache
```

目前仓库同时保留 Markdown 和 PDF 两种形式。Markdown 适合在线检索、引用和继续修改；PDF 通常是论文原文、译文排版稿或离线阅读版本。阅读时优先从下面列出的 Markdown 入口开始。

## 存储领域高手

这一部分不是简单罗列文件系统名词，而是试图建立一套分析存储系统的视角：数据从哪里来，经过哪些层，元数据如何组织，故障和一致性如何处理，性能瓶颈又出现在什么位置。

### 1. 文件存储与数据路径

先理解文件系统提供了什么抽象，以及一次读写在主机内部如何移动。

- [漫谈文件存储](<存储领域高手/漫谈文件存储.md>)：从 GFS 出发，理解 Namespace、Control Flow、Data Flow 和分布式文件系统的基本设计。
- [Linux IO 模型详解](<存储领域高手/Linux IO模型详解.md>)：梳理 POSIX I/O、异步 I/O、io_uring、DPDK 和 SPDK 等路径及其开销边界。
- [传统文件传输发展历程](<存储领域高手/传统文件传输发展历程.md>)：从 `read()` + `send()` 到 `sendfile()`、`splice()`、`MSG_ZEROCOPY` 和 Scatter-Gather DMA。
- [FUSE 模块学习](<存储领域高手/Fuse模块学习.md>)：从 VFS、NFS 讲到用户态文件系统，以及 JuiceFS 为什么选择 FUSE。

### 2. 分布式存储、元数据与一致性

这一组文章关注文件系统之外更通用的系统问题：如何管理海量 Namespace，如何降低元数据竞争，如何在可用性、一致性和性能之间取舍。

- [TafDB 设计思路](<存储领域高手/TafDB设计思路.md>)：从层级 Namespace、平坦 Namespace 到元数据底座的技术演进。
- [如何将千亿文件放进一个文件系统：CFS 论文背后的故事](<存储领域高手/如何将千亿文件放进一个文件系统，EuroSys'23 CFS 论文背后的故事.md>)：以 CFS 为例理解大规模元数据服务的设计动机。
- [CFS 论文译文](<存储领域高手/CFS译文.md>)：配合论文原文阅读 CFS 的架构、分区、事务和垃圾回收设计。
- [Mantle 论文译文](<存储领域高手/Mantle译文md.md>)：理解云对象存储中的层级元数据管理、路径查找和目录更新竞争。
- [Raft 必备的优化手段：Leader Election](<存储领域高手/Raft 必备的优化手段（一）Leader Election 篇.md>)：从 ReadIndex、LeaseRead、PreVote 等机制理解共识系统的工程优化。

### 3. 工程系统与生产案例

这些文章把抽象的设计原则放回真实系统，适合在掌握基础概念后阅读。

- [ByteFUSE 分布式文件系统的演进与落地](<存储领域高手/ByteFUSE分布式文件系统的演进与落地.md>)：观察一个文件系统如何从基础功能走向云原生、稳定性和性能优化。
- [百度 Aries 的设计与思考](<存储领域高手/百度Aries的设计与思考.md>)：从 EB 级数据面系统理解规模、可靠性和工程取舍。
- [字节跳动 EB 级日志系统设计与优化实践](<存储领域高手/字节跳动 EB 级日志系统设计与优化实践.md>)：从采集、存储、索引、查询和成本角度观察大规模日志系统的演进。
- [多模态“卷王”阶跃星辰：JuiceFS 大模型存储平台实践](<存储领域高手/多模态“卷王”阶跃星辰：如何利用 JuiceFS 打造高效经济的大模型存储平台.md>)：观察文件存储如何服务模型分发、缓存和多云推理部署。

### 4. 高性能存储与 AI 场景

这里是存储领域通向 AI Infra 的连接处：数据不只要被保存，还要以足够低的延迟和足够高的带宽送到计算设备。

- [RDMA 学习](<存储领域高手/RDMA学习.md>)：从 TCP 数据路径、内存注册和 Verbs 软件架构理解 RDMA。
- [解锁 RDMA 技术：从原理到应用](<存储领域高手/解锁RDMA 技术：从原理到应用的深度剖析.md>)：补充 RDMA 的通信过程、协议形态和典型应用。
- [3FS 文件系统详解](<存储领域高手/3FS文件系统详解.md>)：从 AI 训练和推理的工作负载出发，理解并行文件系统、SSD 和 RDMA 的组合。
- [DeepSeek 3FS：端到端无缓存的存储新范式](<存储领域高手/DeepSeek 3FS：端到端无缓存的存储新范式.md>)：讨论面向 AI 大文件场景的架构取舍，以及“端到端无缓存”的设计思路。
- [HiCache 官方文档](<存储领域高手/Hicache官方文档.md>)：理解 GPU、主机内存和分布式存储组成的多级 KV Cache。

## 从存储过渡到 AI Infra

进入 AI Infra 后，问题的名字变了，但系统的基本矛盾并没有消失：计算越来越快，数据搬运、内存容量、通信延迟、资源调度和故障处理越来越重要。

| 存储领域的概念 | AI Infra 中的对应问题 |
| --- | --- |
| 文件块、Extent、内存池 | KV Cache Block、PagedAttention、显存分配 |
| Namespace 与元数据服务 | 请求状态、Prefix Cache 索引、分布式 KV Cache 元数据 |
| Page Cache、分层存储、冷热分层 | GPU/CPU/SSD 多级缓存与 KV Cache 迁移 |
| Control Flow 与 Data Flow | Scheduler、Prefill/Decode 调度与 KV Cache 传输 |
| RDMA、DMA、零拷贝 | GPU 间、节点间的高带宽低延迟数据搬运 |
| 副本、故障检测、重试 | 分布式推理中的容错、重试和服务降级 |

理解这条对应关系，比单独记忆某个框架的 API 更重要。它能帮助读者把 vLLM、SGLang、Mooncake、HiCache、3FS 等系统放进同一张架构图中。

## AI infra

这一部分按照“模型基础 → 计算优化 → 训练并行 → 推理内存 → 推理系统 → 分布式服务”的顺序组织。

### 1. 模型与计算基础

- [Transformer 深入浅出介绍](<AI infra/Transformer深入浅出介绍.md>)：从注意力机制和 Transformer 的整体结构开始。
- [一文了解 Attention：从 MHA 到 DeepSeek MLA](<AI infra/一文了解Attention，从MHA到DeepSeek MLA.md>)：梳理 MHA、MQA、GQA、MLA 等注意力变体及其内存影响。
- [从啥也不会到 CUDA GEMM 优化](<AI infra/从啥也不会到CUDA GEMM优化.md>)：从矩阵乘法和 GPU 执行模型理解算子性能优化。

### 2. 大模型训练并行

这组文章回答一个基础问题：当模型、数据或单次训练任务超过单卡能力时，如何把计算和状态分布到多张 GPU 上。

- [数据并行：DP、DDP 与 ZeRO](<AI infra/图解大模型训练之：数据并行(DP, DDP与ZeRO).md>)
- [张量模型并行：TP 与 Megatron-LM](<AI infra/图解大模型训练之：张量模型并行(TP)，Megatron-LM.md>)
- [流水线并行：以 Gpipe 为例](<AI infra/图解大模型训练之：流水线并行（Pipeline Parallelism），以Gpipe为例.md>)

### 3. 推理内存与 vLLM

LLM 推理的关键并不只是把模型前向计算跑起来，还要持续管理不同长度请求的 KV Cache，并在有限显存中维持高吞吐。

- [PagedAttention 译文](<AI infra/PagedAttention译文.md>)：理解分页式 KV Cache 管理的动机、结构和收益。
- [图解大模型计算加速系列：vLLM 源码解析 1、2](<AI infra/图解大模型计算加速系列：vLLM源码解析1，2.md>)：从 LLMEngine、请求生命周期和 Scheduler 进入 vLLM。
- [vLLM 源码解析 3、4](<AI infra/vLLM源码解析3，4.md>)：继续分析 BlockManager、块分配、Prefix Caching 和执行流程。
- [nano-vLLM 分析](<AI infra/nano—vLLM分析.md>)：从一个尽量精简的实现串起调度器、页式 KV Cache、模型执行、Attention 和张量并行。

### 4. 分布式推理、KV Cache 与高性能网络

当单机显存和算力不足以支撑目标负载时，推理系统会进一步拆分 Prefill 与 Decode，跨节点共享或搬运 KV Cache，并依赖高性能网络。

- [PD（Prefill & Decode）分离概述](<AI infra/PD(Prefill&Decode)分离概述.md>)：理解 Prefill 和 Decode 的不同瓶颈，以及为什么要进行解耦。
- [Mooncake 译文](<AI infra/Mooncake译文.md>)：理解以 KV Cache 为中心的分布式推理架构，以及用更多存储换取更少计算的思路。
- [Mooncake 译文 2](<AI infra/Mooncake译文2.md>)：补充 Mooncake 的系统设计与实现细节。
- [RDMA 技术架构深度解析](<AI infra/RDMA技术架构深度解析.md>)：从主机内存、网卡、DMA 和协议路径理解 AI 集群中的高性能通信。
- [InfiniBand vs RoCE 详解](<AI infra/InfiniBand vs RoCE详解.md>)：比较两类高性能网络的协议、流控、部署和运维特点。
- [NVIDIA Dynamo：分布式 AI 推理引擎](<AI infra/NVIDIA Dynamo 分布式AI推理的高效引擎.md>)：观察生产级分布式推理系统如何组合调度、路由、KV Cache 和传输能力。

这一组内容可以和存储目录中的 [3FS](<存储领域高手/3FS文件系统详解.md>)、[HiCache](<存储领域高手/Hicache官方文档.md>)、[RDMA 学习](<存储领域高手/RDMA学习.md>) 交叉阅读。

## PDF 与论文资料

仓库中还保留了不少论文原文、会议材料和译文 PDF，例如：

- [DistServe：Disaggregating Prefill and Decoding](<AI infra/DistServe——Disaggregating Prefill and Decoding.pdf>)
- [PD 分离与 Chunked Prefill](<AI infra/PD分离与Chunked Prefill.pdf>)
- [Prefix Caching：实现 KV Cache 的跨请求高效复用](<AI infra/Prefix Caching 详解：实现 KV Cache 的跨请求高效复用.pdf>)
- [RocketKV：Accelerating Long-Context LLM Inference](<AI infra/RocketKV——Accelerating Long-Context LLM Inference.pdf>)
- [Speculative Decoding 推测解码方案详解](<AI infra/Speculative Decoding 推测解码方案详解.pdf>)
- [FlashAttention](<AI infra/FlashAttetion.pdf>) 与 [FlashDecoding++](<AI infra/FlashDecoding++Next.pdf>)

PDF 适合保存原始材料或离线阅读，具体观点和结论仍应回到论文原文、项目源码和官方文档核对。

## 这份仓库适合怎样使用

- 把 README 当作地图，把具体文章当作专题入口，不必按文件名逐个打开。
- 阅读一篇文章时，优先画出它的数据路径、控制路径、状态和资源边界。
- 阅读论文译文后，再结合原文的设计目标、实验设置和限制条件判断结论能否迁移。
- 阅读 vLLM、Mooncake 或 3FS 时，主动把它们与已经掌握的文件系统、缓存、I/O 和网络概念联系起来。

## 内容状态

这是持续整理中的个人学习仓库。文章的成熟度并不完全一致：有些是系统化笔记，有些是论文译文，有些是源码阅读记录或工程案例。笔记中的观点、链接和外部图片也可能随着项目版本和资料更新而变化，欢迎通过 Issue 或 Pull Request 指出错误、补充资料和改进目录。

