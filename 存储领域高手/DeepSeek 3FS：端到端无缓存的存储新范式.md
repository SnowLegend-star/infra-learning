# DeepSeek 3FS：端到端无缓存的存储新范式

在 2025 年 2 月 28 日，DeepSeek 正式开源了其高性能分布式文件系统 3FS**【1】**，作为其开源周的压轴项目，3FS 一经发布便引发了技术圈的热烈讨论。它不仅继承了分布式存储的经典设计，还通过极简却高效的架构，展现了存储技术的新趋势。

作为一名存储从业者，我将以分布式存储架构的视角，深入剖析 3FS 的设计亮点，揭示其“**端到端无缓存**”理念背后的技术演进。

## **架构全貌**

3FS 采用典型的分布式文件系统架构，客户端直接连接元数据和数据服务，整体设计清晰且高效。其核心组件包括：

- **元数据集群**：元数据服务负责将文件语义转化为键值（KV）语义，底层选用 FoundationDB 作为 KV 存储。FoundationDB 以高可用性和数据冗余著称，这种设计与 JuiceFS、CephFS 等主流文件系统类似，确保元数据的可靠性和一致性。
- **数据集群**：数据高可靠性通过链式复制协议实现，这一协议最早在 Azure Block Storage 中得到验证，以读优化为特色。后来 Meta 的 Delta 存储系统也应用了该复制协议。目前 3FS 支持多副本存储，虽然配置文件中提及纠删码（EC），但代码中尚未实现，可能是功能尚未完善或未纳入开源计划。但链式复制带来的写入延迟较高，需要在上层尽可能带宽以大 IO 为主。
- 集群管理采用了 ClickHouse 作为监控数据存储，通过心跳管理来实现故障检测。

![图片](https://mmbiz.qpic.cn/mmbiz_png/sWu6ZDL8Sd4Zlub9RDKhJmvTwuvcmBJCYBWMXQbM6hlsBj7duUticVUIwGo0jW762FKhLicdJaglElqvcqRjNWyA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

对于存储从业者来说，3FS 的架构选型既有值得称道之处，也存在潜在的技术弱点：

- FoundationDB 的选择：FoundationDB 以事务一致性和高可用性见长，但并非以极致性能著称。相比 Lustre 或 GPFS 等并行文件系统，3FS 的元数据性能扩展性受限于 FoundationDB 的能力，在高并发场景下可能面临瓶颈。
- 元数据延迟隐患：**文件元数据与 KV 存储形成两个平面**，会产生额外的网络转发和处理开销。对于小文件读写密集的场景，这种设计可能显著增加延迟。

不过，3FS 明确将目标锁定在 AI 相关场景，专注于大文件和吞吐带宽优化，因此小文件性能的短板被有意规避，其设计更像是一场“取舍的艺术”。

## **端到端无缓存：存储架构设计新范式**

3FS 项目开篇就讲到了整个项目是面向 AI 相关场景设计，完全面向大文件优化，放弃对于传统文件系统小文件问题的掣肘。因此元数据并发性能和延迟的劣势可以稍稍避开，而 3FS 相对于过去并行文件系统和分布式 NAS 到底有何架构上的差异？答案就是“端到端无缓存”。

讲述端到端无缓存，我们需要首先回到存储系统早期最重要的架构选择，分层存储和缓存系统设计上。

在存储系统发展的早期，DRAM 缓存加 HDD 是主流方案，缓存设计弥补了硬件性能的巨大差距。随着 SSD 的引入，DRAM+SSD 的组合成为优化热点数据的利器。然而，全闪存时代的到来让复杂的缓存机制逐渐成为瓶颈。NVMe SSD 的超高性能，使得 DRAM 缓存在某些场景下反而拖慢了系统效率。

“去元数据缓存”兴起

而到了 2014 年开始，随着全闪文件系统的提出和概念产品的落地，业界逐渐意识到复杂的元数据和数据缓存机制影响了 SSD 性能的充分发挥。随着 SSD 介质的迭代和 PCIe 标准的演进，NVMe SSD 的性能进一步提升，在相当多的场景里，DRAM 缓存设计已经成为了系统瓶颈，最早是在存储系统的数据缓存中，如 NetApp Ontap 系统：

- NetApp 在 FAST 2024 上的《Evolution of ONTAP to Low-Latency SSDs》**【2】**论文，提及如何面对一个老旧的架构，通过 Topspin Read 和 Ack Early 来绕开冗长的 IO 栈。

然后是新一代文件系统如 VastData、WekaFS、DAOS，都在元数据存储上彻底放弃了 DRAM 缓存：

- VastData 的 DASE（Disaggregated Shared-Everything Architecture）架构最早著名的设计就是借助于 Intel Optane 介质，通过持久化内存（PMem）形态去解决 DRAM 的掉电问题，同时将 PMem 作为元数据存储和写入缓冲，降低相对于 TLC 介质的读写延迟。
- WekaFS 则采用文件系统命名空间彻底打散的方式，完全借助于分布式集群的横向扩展能力，使用上千个 MDS 逻辑分区并用大量 NVMe SSD 并发来替代集中式的 DRAM 性能。
- DAOS 与 VastData DASE 架构类似，也采用了 Intel Optane 作为元数据引擎存储介质，但是随着 Optane 介质下市，DAOS 还在艰难转型 TLC SSD 方案。

以上都是在 2014年-2020 年，整个高性能文件系统领域呈现的元数据无缓存设计趋势。

3FS 的元数据采用了 FoundationDB 作为底座，在客户端侧放弃了 Posix 要求的元数据一致性要求和传统并行客户端的多客户端一致性，相比上述几家领先的元数据引擎避开了元数据性能的巨大挑战，但是这个避让不是单方面的，而是借助于 **FFRecord**。

3FS 的小文件答案：FFRecord

过去通用存储系统架构师仅仅站在存储协议上考虑，最大的挑战就是无法避免业务场景的小文件，而 DeepSeek 作为端到端系统工程的卓越代表，通过提供 FFRecord 来极大解决了业务场景里潜在的小文件情况。

FFRecord 是由 DeepSeek 开发的一种文件格式，专门用于高效存储和访问二进制记录序列，特别适用于深度学习（DL）训练样本。具有以下关键特性：

- 随机访问：支持直接访问特定记录，无需扫描整个文件。
- 异步 I/O（AIO）：基于 Linux AIO，提供非阻塞数据读取，提升数据加载效率。
- 高效批量读取：能够快速读取数据批次，适合深度学习中批量处理的需求。
- 压缩支持：可选的记录级压缩，节省存储空间，同时保持高效访问。

在实际的训练中，FFRecord 可以将训练样本按记录存储（如图像数据集中的每张图片为一条记录），而 3FS 的分布式特性确保这些文件可以跨多个节点存储，适合大规模数据集。

FFRecord 支持随机访问和批量读取，而 3FS 的无缓存设计确保数据直接从存储介质传输到应用程序，避免了不必要的中介层。这种结合减少了数据访问延迟。

FFRecord 的异步读取功能（基于 Linux AIO）与 3FS 的高吞吐量架构相辅相成。在分布式训练中，多个节点可以同时发起异步请求，从 3FS 获取 FFRecord 数据，从而加速数据加载。训练框架（如 PyTorch 或 TensorFlow）可以直接读取 FFRecord 格式的数据，无需额外的格式转换或缓存管理。

通过 FFRecord，3FS 在应用侧提供一定的场景解决方案，错位避开了传统并行文件系统的小文件难题。

从 FUSE 优化到“无客户端缓存” 

3FS 系统的架构师更早看到了 GPU 和 CPU 对于存储性能访问需求的变化，非常早的采用了端到端无缓存设计，即“从无数据缓存->无元数据缓存->无客户端缓存”。

这里就需要提到 3FS 的客户端技术选型和设计，3FS 客户端采用了主流 FUSE 框架，对于 FUSE 每个文件系统从业者都是爱恨交加，一方面 FUSE 给 Linux Kernel FS 带来了用户态实现的可能性，另一方面糟糕的性能实现以及坎坷的演进使得每个 FUSE 使用者都无力吐槽。

但也不得不说，FUSE 在过去 3 年里，随着文件系统在 AI 场景的使用，大量的 FUSE 改进项目和内核优化都在进行中，其中包括以下：

- **XFUSE**: 《XFUSE: An Infrastructure for Running Filesystem Services in User Space 》**【3】**在 2021 的 ATC，阿里云就提出了 FUSE 存在的若干性能问题，对 FUSE 进行了多方面优化，包括路径直通、批处理请求、多线程处理等。论文结果表明，XFUSE 能将用户态文件系统请求处理延迟压缩到 4 微秒级，吞吐达到 8GB/s，同时，XFUSE 保持了对 FUSE API 的兼容，方便现有 FUSE 文件系统迁移。
- **ByteFUSE**：字节跳动文件系统团队在 2021 年开始也在逐步优化 FUSE 性能，在《[ByteFUSE分布式文件系统的演进与落地](https://mp.weixin.qq.com/s?__biz=Mzg3Mjg2NjU4NA==&mid=2247484577&idx=1&sn=acf23f871cb1805b8fbdecfd2ddb7ab9&scene=21#wechat_redirect)》可以看到其创新性的提出了利用 VirtQueue 来优化队列性能，并且可以统一虚拟机和容器场景。VirtQueue 作为 QEMU/KVM 成熟的高性能队列框架，非常适合去解决 FUSE 的队列问题。
- **RFuse**：《RFUSE: Modernizing Userspace Filesystem Framework through Scalable Kernel-Userspace Communication》**【4】**这篇 FAST 24 的论文进一步对 FUSE 通道进行性能优化，它的核心思路是使用每核独立的环形缓冲区在内核和用户态之间传递消息，从而避免传统 FUSE 中集中队列和锁带来的瓶颈。每个 CPU 核心都有自己的请求通道，用户态文件系统可以多线程并行处理不同核心的请求，无需在内核模块中进行序列化。更重要的是，RFUSE 设计为与现有 FUSE 文件系统兼容，不需要修改现有用户态代码。
- **Fuse over io_uring**：由 DDN Bernd Schubert 在 2023 年提出，在 2024 年经过近 10 个版本的迭代，由于代码涉及 CPU 调度和跨模块的影响，被迫砍掉了若干性能优化点，在 2025 年合并进了主线。Fuse over io_uring **【5】**特性原理是用 io_uring 取代传统 FUSE 的 /dev/fuse 通道，这样用户态文件系统可以直接从 io_uring 队列读取请求和提交响应，避免了每次请求的系统调用和上下文切换。

3FS 采用 FUSE 也同样需要面对类似的问题，不过并不像上述的 FUSE 改进计划局限在 FUSE 内核里，3FS 是选择完全绕开 FUSE 的数据读写通道，如同 DeepSeek 过去开源的算子优化类似，通过借鉴 io_uring 的零拷贝和共享内存设计，直接在应用侧跟文件客户端建立了共享内存通道，实现从应用侧内存数据到 RDMA 传输的零拷贝，更是彻底避免了 VFS 系统调用的 Context Switch 和内存拷贝。

![图片](https://mmbiz.qpic.cn/mmbiz_png/sWu6ZDL8Sd4Zlub9RDKhJmvTwuvcmBJCnrYxo3f1WYtwU76xkC5pMcdKp4WnHxMI2rc9Vx4lTFHicTibWYDZD9Qg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1)

这个设计的实现最早可以追溯到 DAOS 的 Dfuse+Libioil **【6】**方案，可惜的是，目前没有看到 DAOS 的这个方案在实际的 AI 开源框架中实践。

实际上，不仅是 DeepSeek 3FS，几个前沿的存储系统都开始在端到端无缓存方向上发力：

- VastData 在 5.1 版本开始，在其两层的 DASE 架构上，进一步加强了 QLC SSD 直写，减少 SCM 层带来的写入带宽瓶颈。
- HammerSpace 在去年年底，创新性提出了 Tier-0 存储概念，即在计算节点侧直接利用本地 NVMe SSD 作为读写，纳入到文件系统的全局里，并且不改变应用的使用方式。其核心是在利用 Linux Kernel 6.12 的 NFS Bypass 特性，绕开整个 VFS 栈进行读写，让 NVMe SSD 在单文件并发上发挥接近块存储的性能。

![图片](https://mmbiz.qpic.cn/mmbiz_png/sWu6ZDL8Sd4Zlub9RDKhJmvTwuvcmBJCY8F54ibdwIFxfQ1X7wM1G3SIWnL32g1Wgau5grFstwNj6U06HUPTJDA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)

XSKY 在去年的英特尔存储技术峰会《[LLM 存储，架构至关重要](https://mp.weixin.qq.com/s?__biz=MzAwNTc0OTM1NA==&mid=2652419216&idx=1&sn=635412eff62a60e944defee3fd47d1a5&scene=21#wechat_redirect)》提出观点，随着 AI LLM 场景对于存储带宽性能的爆发，带宽性能成为文件系统存储性能的绝对优先级，而无缓存设计是其中关键。

XSKY 星飞平台自 2023 年底发布，坚持“全共享架构”、“端到端 NVMe”和“单层闪存介质”三大设计原则，在新时代的 NVMe + RDMA 高性能方向上，彻底放弃架构中的缓存层，完全使用单层闪存介质来满足。跟 3FS 的端到端无缓存思路不谋而合。

总而言之，如同过去系统工程领域著名的口号“no caches， no locks”，3FS 的实践证明，在全闪存系统里，通过架构和算法可以弥补不用缓存可能带来的问题，反而获得更简洁高效的系统。

## **3FS 与开源 KVCache 框架**

3FS 不仅仅用于训练侧，其设计特点也非常有助于在推理 LLM 兴起后，用于大规模 KVCache 场景。正如 DeepSeek R1 模型发布后，《[DeepSeek 引领的 AI 范式转变与存储架构的演进](https://mp.weixin.qq.com/s?__biz=MzAwNTc0OTM1NA==&mid=2652424348&idx=1&sn=113e4df0ef61a27c0ee13486399a09fd&scene=21#wechat_redirect)》的推理过程数据处理和存储场景要求，以及 DeepSeek 发表的《Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention》**【7】**和 Kimi 在 FAST 25 的《Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot》**【8】**，笔者非常期待 Mooncake 项目和 3FS 项目的整合，3FS 为 Mooncake 提供可靠的 KVCache 共享池，使得开源领域里出现一个完整的开源 KVCache 方案。

值得一提的是，DeepSeek 所实践的 Prefill-Decode 分离正在通过 Kimi/Alibaba 工程师的努力进入 vLLM 社区，PD 分离将是大规模分布式推理服务和高性能存储结合的起点，Mooncake 论文表明共享高性能存储方案会比 Local Cache 方案缓存命中率提升 2.6x，响应时间降低 48%。3FS 可以极大帮助超长上下文 Prefill 阶段的 KVCache 加载。

## **3FS 与云数仓**

这次跟 3FS 共同推出的还有 Smallpond**【9】**，一个基于 DuckDB 构建，将 3FS 作为持久化的轻量级数据处理框架。过去分布式 OLAP 数据库从存算一体转向存算分离，从 ClickHouse 为代表转向一众基于 S3 共享存储的云数仓。但是这个转向最大的挑战就是如何构建缓存层，目前不管是著名的 Snowflake 还是开源的 Databend 云数仓，都是在计算层构建缓存，3FS 是否会给云数仓新的缓存机会，特别是计算、缓存、持久化全分离的模式，比如在本地机房构建多个计算集群，但是共享同一个缓存集群，然后将持久化数据放在远端。

DuckDB 近日正式提出了外部文件缓存概念**【10】**，笔者认为是一个改变云数仓架构演进的尝试，标准化外部缓存能力，不同计算架构和集群共享同一个缓存集群有可能成为新的分离式趋势。

这些可能的变化都让笔者想起了 20 年前 Google 发布的 GFS 论文，特别是后来又有开源的 Apache HDFS，我们期待 3FS 可以成为 AI 领域的 “HDFS”。

## **未来展望**

通过上述对 3FS 的无缓存设计解析，我们可以看到 3FS 在架构上大胆地结合了多种前沿理念，成为当前存储技术趋势的一个缩影和引领者：

- **融合创新：**3FS 身上融合了来自分布式数据库、对象存储和高性能计算领域的创新：FoundationDB 提供强一致元数据存储，链式复制保障高可靠数据冗余，无缓存架构和 RDMA 加速满足极致性能需求，用户态文件系统优化兼顾了灵活性和效率。可以说，3FS 并非孤立发明每一个轮子，而是巧妙地把“业界最佳实践”集成到一个系统中。例如，从 Delta、Azure 借鉴链式复制的可靠性模型，从 WekaFS、Vast Data 全闪存直写的简洁路径，从 Alibaba、FAST 论文的 FUSE 提速的思路，再加以自身的工程实现，打造出了这样一个系统。
- **技术趋势的验证者**：3FS 的出现也在一定程度上验证并推动了存储技术的发展趋势。过去几年业内对无缓存、用户态 IO、RDMA、分布式事务元数据等理念多有探讨，3FS 将它们付诸实践并取得成功案例，会让更多人意识到这些路线的可行性。例如，一些保守的企业存储厂商可能尚在观望无缓存架构是否可靠，而 3FS 的性能和稳定性如果获得认可，将促使整个行业更大胆地拥抱这一趋势。

当然，3FS 作为新晋系统，未来仍有许多挑战和发展空间。例如，在更大规模部署下，其 FoundationDB 元数据层的伸缩性、链式复制对网络的压力、无缓存设计在某些工作负载下的权衡，都需要通过实践进一步观察。同时，内核社区对 FUSE 优化也将为 3FS 作为通用文件客户端的性能提升带来机遇，以及 3FS 支持 GPU 直通存储的可能性，在生态上，3FS 未来是否能够进入主流 AI 框架的默认引擎将是值得观察和期待的信号。

可以预见，存储系统架构在全闪存和新型网络时代会加速演进，而 3FS 作为这一潮流中的先锋，将不断优化并引入新的特性。总的来说，3FS 的评测让我们看到了未来存储系统的一种雏形：软件更“通透”但更智能，硬件性能得到充分释放，分布式一致性不再以牺牲效率为代价。对于广大的存储技术爱好者而言，这是一个激动人心的方向。我们期待 3FS 在后续版本中继续打磨，实现更成熟的功能，并与业界同行一起，将现代存储架构带入一个新的境界。



**注：**

【1】DeepSeek  3FS

https://github.com/deepseek-ai/3FS

【2】《Evolution of ONTAP to Low-Latency SSDs》

https://www.usenix.org/system/files/fast24-curtis-maury.pdf

【3】《XFUSE: An Infrastructure for Running Filesystem Services in User Space》

https://www.usenix.org/system/files/atc21-huai.pdf

【4】《RFUSE: Modernizing Userspace Filesystem Framework through Scalable Kernel-Userspace Communication》

 https://www.usenix.org/system/files/fast24-cho.pdf

【5】Fuse over io-uring 

https://www.phoronix.com/news/Linux-6.14-FUSE

【6】 Dfuse+Libioil

https://www.intel.com/content/www/us/en/developer/articles/training/introduction-to-dfs.html

【7】《Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention》

https://arxiv.org/abs/2502.11089

【8】 Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot

https://www.usenix.org/system/files/fast25-qin.pdf

【9】 Smallpod

https://github.com/deepseek-ai/smallpond

【10】DuckDB 外部文件缓存

https://github.com/duckdb/duckdb/pull/16463