# RDMA技术架构深度解析

`RDMA`，即 `Remote Direct Memory Access`，是一种绕过**远程**主机 `OS kernel` 访问其内存中数据的技术，概念源自于 `DMA` 技术。在 `DMA` 技术中，外部设备（`PCIe` 设备）能够绕过 `CPU` 直接访问 `host memory`；而 `RDMA` 则是指外部设备能够绕过 `CPU`，不仅可以访问本地主机的内存，还能够访问另一台主机上的用户态内存。由于不经过操作系统，不仅节省了大量 `CPU` 资源，同样也**提高了系统吞吐量**、**降低了系统的网络通信延迟**，在高性能计算和深度学习训练中得到了广泛的应用。

## 技术背景

计算机网络通信中最重要两个衡量指标主要是 **带宽** 和 **延迟**。 现实计算机网络中的通信场景中，主要是以发送小消息为主，因此处理延迟是提升性能的关键。 传统的`TCP/IP`网络通信，数据需要通过用户空间发送到远程机器的用户空间，在这个过程中需要经历若干次内存拷贝：

![alt text](https://johng.cn/assets/images/image-fd44c1a3f2250bdbe330584739f2ec74.png)

- 数据发送方需要讲数据从用户空间`Buffer`复制到内核空间的`Socket Buffer`
- 数据发送方要在内核空间中添加数据包头，进行数据封装
- 数据从内核空间的`Socket Buffer`复制到`NIC Buffer`进行网络传输
- 数据接受方接收到从远程机器发送的数据包后，要将数据包从`NIC Buffer`中复制到内核空间的`Socket Buffer`
- 经过一系列的多层网络协议进行数据包的解析工作，解析后的数据从内核空间的`Socket Buffer`被复制到用户空间`Buffer`
- 这个时候再进行系统上下文切换，用户应用程序才被调用

在高速网络条件下，传统的`TCP/IP`网络在**主机侧数据移动和复制操作带来的高开销**限制了可以在机器之间发送的带宽。为了提高数据传输带宽，人们提出了多种解决方案，这里主要介绍下面两种：

- `TCP Offloading Engine`
- `Remote Direct Memroy Access`

## TCP Offloading Engine

在主机通过网络进行通信的过程中，`CPU` 需要耗费大量资源进行多层网络协议的数据包处理工作，包括数据复制、协议处理和中断处理。当主机收到网络数据包时，会引发大量的网络 `I/O` 中断，`CPU` 需要对 `I/O` 中断信号进行响应和确认。为了将 `CPU` 从这些操作中解放出来，人们发明了`TOE`（`TCP/IP Offloading Engine`）技术，将上述主机处理器的工作转移到网卡上。`TOE` 技术需要特定支持 `Offloading` 的网卡，这种特定网卡能够支持封装多层网络协议的数据包。

![alt text](https://johng.cn/assets/images/image-1-b162f1c128b495c6ac1859c8d8bbde8c.png)

- `TOE` 技术将原来在协议栈中进行的`IP`分片、`TCP`分段、重组、`checksum`校验等操作，转移到网卡硬件中进行，降低系统`CPU`的消耗，提高服务器处理性能。
- 普通网卡处理每个数据包都要触发一次中断，**`TOE` 网卡则让每个应用程序完成一次完整的数据处理进程后才触发一次中断**，显著减轻服务器对中断的响应负担。
- `TOE` 网卡在接收数据时，在网卡内进行协议处理，因此，它不必将数据复制到内核空间缓冲区，而是直接复制到用户空间的缓冲区，这种"零拷贝"方式避免了网卡和服务器间的不必要的数据

## RDMA(Remote Direct Memroy Access)

为了消除传统网络通信带给计算任务的瓶颈，我们希望更快和更轻量级的网络通信，由此提出了`RDMA`技术。`RDMA`利用`Kernel Bypass`和 `Zero Copy`技术提供了低延迟的特性，同时减少了`CPU`占用，减少了内存带宽瓶颈，提供了很高的带宽利用率。`RDMA`提供了给基于 `IO` 的通道，这种通道允许一个应用程序通过`RDMA`设备对远程的虚拟内存进行直接的读写。

`RDMA` 技术有以下几个特点：

- **CPU Offload**：无需`CPU`干预，应用程序可以访问远程主机内存而不消耗远程主机中的任何`CPU`。远程主机内存能够被读取而不需要远程主机上的进程（或`CPU`)参与。远程主机的`CPU`的缓存(`cache`)不会被访问的内存内容所填充
- **Kernel Bypass**：`RDMA` 提供一个专有的 `Verbs interface`而不是传统的`TCP/IP Socket interface`。**应用程序可以直接在用户态执行数据传输**，不需要在内核态与用户态之间做上下文切换
- **Zero Copy**：每个应用程序都能直接访问集群中的设备的虚拟内存，这意味着应用程序能够直接执行数据传输，在不涉及到网络软件栈的情况下，数据能够被直接发送到缓冲区或者能够直接从缓冲区里接收，而不需要被复制到网络层。

> 假设有两个节点A和B，节点A想要发送一条消息到节点B。
>
> 传统TCP/IP通信流程（非RDMA）：
>
> 1. 节点A：
>    - 应用程序将数据从应用缓冲区拷贝到内核缓冲区（一次拷贝）。
>    - 内核协议栈处理数据（构建TCP/IP包等）。
>    - 数据从内核缓冲区拷贝到网络适配器的缓冲区（第二次拷贝）。
>    - 网络适配器发送数据。
> 2. 节点B：
>    - 网络适配器接收到数据，拷贝到内核缓冲区（第一次拷贝）。
>    - 内核协议栈处理数据（验证校验和、解封装等）。
>    - 数据从内核缓冲区拷贝到应用程序缓冲区（第二次拷贝）。
>
> 使用RDMA的通信流程：
>
> 1. 节点A：
>    - 应用程序直接将要发送的数据的地址和长度等信息提供给RDMA网卡（不需要拷贝数据，零拷贝）。
>    - RDMA网卡直接从用户空间缓冲区读取数据并发送（不需要内核参与，内核旁路）。
> 2. 节点B：
>    - RDMA网卡直接将接收到的数据写入到应用程序预先注册的缓冲区中（零拷贝，不需要内核参与，内核旁路）。

下面是 `RDMA` 整体框架架构图，从图中可以看出，`RDMA`在应用程序用户空间，提供了一系列 `Verbs` 接口操作`RDMA`硬件。`RDMA`绕过内核直接从用户空间访问`RDMA` 网卡。`RNIC`网卡中包括 `Cached Page Table Entry`，用来将虚拟页面映射到相应的物理页面。

![alt text](https://johng.cn/assets/images/image-2-9b5d6573ea020b62dda53f3626c485a6.png)

目前`RDMA`有三种不同的硬件架构实现，它们都可以使用同一套`API`来使用，但它们有着不同的物理层和链路层：

- **Infiniband：** 基于 `InfiniBand` 架构的 `RDMA` 技术，由 `IBTA`（`InfiniBand Trade Association`）提出。搭建基于 `IB` 技术的 `RDMA` 网络需要专用的 `IB` 网卡和 `IB` 交换机。从性能上，很明显`Infiniband`网络最好，但网卡和交换机是价格也很高，然而`RoCEv2`和`iWARP`仅需使用特殊的网卡就可以了，价格也相对便宜很多。
- **iWARP：** `Internet Wide Area RDMA Protocal`，基于 `TCP/IP` 协议的 `RDMA` 技术，由 `IETF` 标 准定义。`iWARP` 支持在标准以太网基础设施上使用 `RDMA` 技术，而不需要交换机支持无损以太网传输，但服务器需要使用支持`iWARP` 的网卡。与此同时，受 `TCP` 影响，性能稍差。
- **RoCE：** 基于以太网的 `RDMA` 技术，也是由 `IBTA` 提出。`RoCE`支持在标准以太网基础设施上使用`RDMA`技术，但是需要交换机支持无损以太网传输，需要服务器使用 `RoCE` 网卡，性能与 `IB` 相当。

![alt text](https://johng.cn/assets/images/image-3-0b2d0c20c1cfdfacb989908890bb6713.png)

### I/O 瓶颈

时间回退到二十世纪的最后一年，随着`CPU`性能的迅猛发展，早在`1992`年`Intel`提出的 [PCI](https://en.wikipedia.org/wiki/Peripheral_Component_Interconnect) 技术已经满足不了人民群众日益增长的`I/O`需求，`I/O` 系统的性能已经成为制约服务器性能的主要矛盾。尽管在`1998`年，`IBM` 联合 `HP` 、`Compaq` 提出了 [PCI-X](https://en.wikipedia.org/wiki/PCI-X) 作为`PCI`技术的扩展升级，将通信带宽提升到`1066 MB/sec`，人们认为`PCI-X` 仍然无法满足高性能服务器性能的要求，要求构建下一代`I/O`架构的呼声此起彼伏。经过一系列角逐，`Infiniband`融合了当时两个竞争的设计 `Future I/O` 和 `Next Generation I/O`，建立了 `Infiniband` 行业联盟，也即 [BTA (InfiniBand Trade Association)](https://www.infinibandta.org/)，包括了当时的各大厂商 `Compaq`、`Dell`、`HP`、`IBM`、`Intel`、`Microsoft` 和 `Sun`。在当时，`InfiniBand` 被视为替换 `PCI` 架构的下一代 `I/O` 架构，并在 `2000` 年发布了 `1.0` 版本的 `Infiniband` 架构 `Specification`，`2001` 年 `Mellanox` 公司推出了支持 `10 Gbit/s`通信速率的设备。

![alt text](https://johng.cn/assets/images/image-4-6bafaaf0d26bba2c474b324097c182a3.png)

然而好景不长，`2000` 年互联网泡沫被戳破，人们对于是否要投资技术上如此跨越的技术产生犹豫。`Intel` 转而宣布要开发自己的 [PCIe](https://en.wikipedia.org/wiki/PCI_Express) 架构，微软也停止了 `IB` 的开发。尽管如此，`Sun` 和 日立等公司仍然坚持对 `InfiniBand` 技术的研发，并由于其强大的性能优势逐渐在集群互联、存储系统、超级计算机内部互联等场景得到广泛应用，其软件协议栈也得到标准化，[Linux 也添加了对于 Infiniband 的支持](https://en.wikipedia.org/wiki/InfiniBand#History)。进入`2010`年代，随着大数据和人工智能的爆发，`InfiniBand` 的应用场景从原来的超算等场景逐步扩散，得到了更加广泛的应用，`InfiniBand` 市场领导者 `Mellanox` 被 `NVIDIA` 收购，另一个主要玩家 `QLogic` 被 `Intel` 收购，`Oracle` 也开始制造自己的 `InfiniBand` 互联芯片和交换单元。到了 `2020` 年代，`Mellanox` 最新发布的 `NDR` 理论有效带宽已经可以达到 [单端口 400 Gb/s](https://www.nvidia.com/en-us/networking/ndr/)，为了运行 `400 Gb/s` 的 `HCA` 可以使用 `PCIe Gen5x16` 或者 `PCIe Gen4x32`。

![alt text](https://johng.cn/assets/images/image-5-9bd9e853bd4457f036f492435f785a32.png)

### 架构组成

`InfiniBand` 架构为系统通信定义了多种设备：`channel adapter`、`switch`、`router`、`subnet manager`，它提供了一种基于通道的点对点消息队列转发模型，每个应用都可通过创建的虚拟通道直接获取本应用的数据消息，无需其他操作系统及协议栈的介入。

![alt text](https://johng.cn/assets/images/image-6-e755cbbcd50e26020395cbff1c9c6c2b.png)

在一个子网中，必须有至少每个节点有一个`channel adapter`，并且有一个`subnet manager` 来管理`Link`。

![alt text](https://johng.cn/assets/images/image-7-0660da567db960113be1738b35f6803d.png)

#### Channel Adapters

可安装在主机或者其他任何系统(如存储设备)上的**网络适配器**，这种组件为数据包的始发地或者目的地，支持`Infiniband` 定义的所有软件 `Verbs`

- `Host Channel Adapter`：`HCA`
- `Target Channel Adapter`：`TCA`

#### Switch

`Switch` 包含多个 `InfiniBand` 端口，它根据每个数据包 `LRH` 里面的 `LID`，负责将一个端口上收到的数据包发送到另一个端口。除了 `Management Packets`，`Switch` 不产生或者消费任何 `Packets`。它包含有 `Subnet Manager` 配置的转发表，能够响应 `Subnet Manager` 的 `Management Packets`。

![alt text](https://johng.cn/assets/images/image-8-ad476c6a913dadd9a80aec66ba28d022.png)

#### Router

`Router` 根据 `L3` 中的 `GRH`，负责将 `Packet` 从一个子网转发到另一个子网，当被转到到另一子网时，`Router` 会重建数据包中的 `LID`。

#### Subnet Manager

`Subnet Manager` 负责配置本地子网，使其保持工作：

- 发现子网的物理拓扑
- 给子网中的每个端口分配`LIC` 和其他属性（如活动`MTU`、活动速度）
- 给子网交换机配置转发表
- 检测拓扑变化（如子网中节点的增删）
- 处理子网中的各种错误

### 分层设计

`InfiniBand` 有着自己的协议栈，从上到下依次包括传输层、网络层、数据链路层和物理层：

![alt text](https://johng.cn/assets/images/image-9-5af9de187a10e1a449a654a6f01f49a0.png)

对应着不同的层，数据包的封装如下，下面将对每一层的封装详细介绍：

![alt text](https://johng.cn/assets/images/image-10-445315b0a1d826bc23e38b6201d5a4ae.png)

#### Physical Layer

物理层定义了 `InfiniBand` 具有的电气和机械特性，`InfiniBand` 支持光纤和铜作为传输介质。在物理层支持不同的 `Link` 速度，每个 `Link` 由四根线组成（每个方向两条），`Link` 可以聚合以提高速率，目前绝大多数的系统采用 `4 Link`。

![alt text](https://johng.cn/assets/images/image-11-f55ebf7e285bc6b84f6a216ede218cd8.png)

以`QDR` 为例，线上的 `Signalling Rate` 为 `10 Gb/s`，由于采用 `8b/10b` 编码，实际有效带宽单 `Link` 为 `10 Gb/s * 8/10 = 8 Gb/s`，如果是 `4 Link`，则带宽可以达到 `32 Gb/s`。因为是双向的，所以 `4 Link` 全双工的速率可以达到 `64 Gb/s`。

![alt text](https://johng.cn/assets/images/image-23-8e98fa41db4a58a73d774d497fe9e1f8.png)

#### Link Layer

`Link Layer` 是 `InfiniBand` 架构的核心，包含以下部分：

- `Packets`：链路层由两种类型的Packets，Data Packet和Management Packet，数据包最大可以为4KB，数据包传输的类型包括两种类型

  - `Memory`：`RDMA read/write，atomic operation`

  - `Channel`：`send/receive，multicast transmission`

- `Switching`：在子网中，Packet的转发和交换是在链路层完成的

  - 一个子网内的每个设备有一个由 `subnet manager` 分配的 `16 bit Local ID` (**LID**)

  - 每个 `Packet` 中有一个 `Local Route Header` (`LRH`) 指定了要发送的目标 `LID`

  - 在一个子网中通过 `LID` 来负责寻址

- `QoS`：链路层提供了QoS保证，不需要数据缓冲

  - `Virtual Lanes`：一种在一条物理链路上创建多条虚拟链路的机制。虚拟通道表示端口的一组用于收发数据包的缓冲区。支持的 `VL` 数是端口的一个属性。

  - 每个 `Link` 支持 `15` 个标准的 `VL` 和一个用于 `Management` 的 `VL15`，`VL15` 具有最高等级，`VL0` 具有最低等级

  - `Service Level`：`InfiniBand` 支持多达 `16` 个服务等级，但是并没有指定每个等级的策略。`InfiniBand` 通过将 `SL` 和 `VL` 映射支持 `QoS`

`Credit Based Flow Control`

`Data Integrity`：链路层通过 `Packet` 中的 `CRC` 字段来进行数据完整性校验，其组成包括 `ICRC` 和 `VCRC`。

![alt text](https://johng.cn/assets/images/image-12-273c2f3925decf63a48adfdf53b2346a.png)

#### Network Layer

网络层负责将 `Packet` 从一个子网路由到另一个子网：

- 在子网间传输的 `Packet` 都有一个 `Gloabl Route Header` (`GRH`)。在这个 `Header` 中包括了该 `Packet` 的 `128 bit` 的 源 `IPv6` 地址和目的 `IPv6` 地址
- 每个设备都有一个全局的 `UID` (`GUID`)，路由器通过每个 `Packet` 的 `GUID` 来实现在不同子网间的转发

下面是 `GRH` 报头的格式，长 `40` 字节，可选，用于组播数据包以及需要穿越多个子网的数据包。它使用 `GID` 描述了源端口和目标端口，其格式与 `IPv6` 报头相同。

<img src="https://johng.cn/assets/images/image-13-6f760296d3aeb9e50c94c350390d941d.png" alt="alt text" style="zoom: 80%;" />

#### Transport Layer

传输层负责 `Packet` 的按序传输、根据 `MTU` 分段和很多传输层的服务(`reliable connection`，`reliable datagram`，`unreliable connection`，`unreliable datagram`，`raw datagram`)。`InfiniBand` 的传输层提供了一个巨大的提升，因为所有的函数都是在硬件中实现的。

![alt text](https://johng.cn/assets/images/image-14-da857838ec8060226a4355ad7f803ee9.png)

按照连接和可靠两个标准，可以划分出下图四种不同的传输模式：

<img src="https://johng.cn/assets/images/image-16-eee06b9a9d475065b2984ba7b1b187b1.png" alt="alt text" style="zoom:67%;" />

- 可靠连接（`RC`）一个 `QP` 只和另一个 `QP` 相连，消息通过一个 `QP` 的发送队列可靠地传输到另一个 `QP` 的接收队列。数据包**按序交付**，`RC` 连接很类似于 `TCP` 连接。
- 不可靠连接（`UC`）一个 `QP` 只和另一个 `QP` 相连，连接是不可靠的，所以数据包可能有丢失。传输层出错的消息不会进行重传，错误处理必须由高层的协议来进行。
- 不可靠数据报（`UD`）一个 `QP` 可以和其它任意的 `UD QP` 进行数据传输和单包数据的接收。不保证按序性和交付性。交付的数据包可能被接收端丢弃。支持多播消息（一对多），`UD` 连接很类似于 `UDP` 连接。

每种模式中可用的操作如下表所示，目前的 `RDMA` 硬件提供一种数据报传输：不可靠的数据报（`UD`），并且不支持 `memory verbs`。

<img src="https://johng.cn/assets/images/image-17-1de1ad002a9bc8183e3ae058946fb676.png" alt="alt text" style="zoom:67%;" />

下面是传输层的 `Base Transport Header` 的结构，长度为 `12` 字节，指定了源 `QP` 和 目标 `QP`、操作、数据包序列号和分区。

![alt text](https://johng.cn/assets/images/image-18-97a2a8a5b5f0859577a17bf972f3f88e.png)

- `Partition Key`：`InfiniBand` 中每个端口 `Device` 都有一个由 `SM` 配置 `P_Key` 表，每个 `QP` 都与这个表中的一个 `P_Key` 索引相关联。只有当两个 `QP` 相关联的 `P_Key` 键值相同时，它们才能互相收发数据包。
- `Destination QP`：`24 bit` 的目标 `QP ID`。

根据传输层的服务类别和操作，有不定长度的扩展传输报头(`Extended Transport Header`，`ETH`)，比如下面是进行时候的 `ETH`：

下面是 `RDMA ETH`，面向于 `RDMA` 操作：

![alt text](https://johng.cn/assets/images/image-19-136cf265bcaae28d111ebc2bf2d2ed16.png)

下面是 `Datagram ETH`，面向与 `UD` 和 `RD` 类型的服务：

![alt text](https://johng.cn/assets/images/image-20-5d5555960dab963581a43c096e37fe86.png)

- `Queue Key`：仅当两个不可靠 `QP` 的 `Q_Key` 相同时，它们才能接受对方的单播或组播消息，用于授权访问目标 `QP` 的 `Queue`。
- `Source QP`：`24 bit` 的 `source QP ID`，用于回复数据包的 `Destination QP`

下面是 `Reliable Datagram ETH`，面向于 `RC` 类型的服务，其中有 `End2End Context` 字段：

![alt text](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA2sAAABbCAIAAABiRYr3AAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAADa6ADAAQAAAABAAAAWwAAAAAgzh+hAAAlG0lEQVR4Ae2debxN1fvHf8g3SmkkMlRCIZlSUoQUFZnKkAapRCFFaTQklKlRE8lcKUOmKDQZGshQSQoVlaIiJBp+72/Pr/Xbr33O2fec69577j7ns/+4d+211957rffeZ+1nPc+znpXn77///h9tIiACIiACIiACIiACIhA3gbxxl1RBERABERABERABERABEfgvAUmQeg9EQAREQAREQAREQAQSIyAJMjFeKi0CIiACIiACIiACIiAJUu+ACIiACIiACIiACIhAYgQkQSbGS6VFQAREQAREQAREQAQkQeodEAEREAEREAEREAERSIyAJMjEeKm0CIiACIiACIiACIiAJEi9AyIgAiIgAiIgAiIgAokROCjO4uXLl1+/fn2chVVMBNKcAIH68+TJk+YQ1PxECei1SZSYyouACGQVgUsuueTVV19N6GrxSpBNmzY999xz+ZvQ1cNYOG/evH/99ZfV/L777itSpMill14axoaozkkkUK9evUWLFiWxArp16AgsXrx47ty5AwYMCF3NVeHcQODdd9+dN2/e/fffnxsqozqEjsDjjz9esWLFRKstK3aixFReBERABERABERABNKdgCTIdH8D1H4REAEREAEREAERSJSAJMhEiam8CIiACIiACIiACKQ7AUmQ6f4GqP0iIAIiIAIiIAIikCgBSZCJElN5ERABERABERABEUh3AvHOxc5VnKpUqbJjx46LLrroiSeeiKzYnDlzbrrpJvLHjRvH/HEr8P777//2229169aNLJ+eOb/++uvw4cOXL1/+ww8/lC1b9vrrr69Tp44Pxc6dO1u1arV169Yrrrji9ttv9x317v74448jR45ctmwZiZIlSzITuVOnTgcffLC3DGnmCb788ssFChR47733fIe0GyICr7322oQJE7788sv//Oc/xPm6+eabK1eubPWP572KbGnAm/bhhx+OGTNm9erVBx10UK1atW677bajjz468grKCS+B559/fsSIEdSf2cSHHXaYa8hdd931/fffu10SxYoVe+CBB7w53vTs2bPHjx+/adOmQw45pEKFCj169ChTpoy3gNKpSoAPytNPPx21dQMHDjzuuOOiHiJA4bBhwz799FNemLPOOotvHImoJZUZi0AoJcivvvrql19++eabb6K2avfu3XQiHELKtL+33HLL2LFjH3vsMUmQRowPNiGKkABs99tvv33rrbcefPDBtm3bWo797dmz52effUZ6z5493nxfevv27YR52rJli+UjkiKYvvPOOy+++CKhkVzhJUuWjBo1ioh3SJAuU4nQEXjyyScHDRrkqr1x48b58+czWmMEEud75c51iVhv2pQpU3r37r1//34ryRvLEOXZZ591JyoRdgKrVq1iYGmPmM7BNYccOhD36C3/5JNPdgV8idGjR/fr189lfvHFF2+88QZDnRNOOMFlKpGqBBhp8Lijtu7uu++Omo/4yGcLacGOIkcihs6aNStfvnxRyyszKoH//8BHPRzGTHoZdJBs1nesXbuWMa63bwpjo7K2ztOnTzfxsUWLFvzAChUqxPURCxwlRPBrr72W/jee+/JFN/GxQYMGffv2PfXUUzmLXyPBydzpu3btQnvkru/ylQgXAZ7j0KFDqfNRRx2F2NexY0cCpxM/1aIYZvheRTY24E3jXv3790eGOOKII3h5GjZsyOm8VOgjI6+jnNAR+PPPPydOnHj11Vf7xERryOeff275WEiq/btZ3xK1pWaPOuaYYxgJd+3alTKMe5977rmohZWZYgSOPfbYMzxbjRo1rIElSpQoWrRo1MY+9dRTJj7eeuutjRs3pswnn3zi/WZFPUuZPgKh1EG6NqD/4KOFCg27NvZWM5seeeSR1atXpwxvFd8nRhVWfunSpeio27VrRzH0YW+//fZ3332HPuzMM8/EIJ5WijF+OcQO3bdvH4Zs1IQrVqwglDFqXXRIhQsXRri84IILonbrjrw3gf6SXQASkvTQQw8tV64ckMlhVGe/TNJIlk5J6T1X6XAR4EdkL8aVV17ZrVs3Ko8OCUMz3/s//vgj+L2KbGnwm4YKyswIyAS8SNu2beOXi0EKd5TISykndAQYFUydOjVWtT/++GM7hBRYunTpWMUsn1cCSwhp3hMzpLzyyit8FzZv3hx8oo6mBoGz/9lcW3iv6JTwscEBxvQj7pAlGF0gOZDG4QoTJR8+vmJkTpo0CWHAV1i7AQRCLEGi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RHm0II6uSVWqVNm8eTNncYqJksidWKvZcH9kKreTSu2UqlWrEqWcC65btw7xsWTJkj6XSndlJURABERABERABERABDJNIHslSF+1ypYt681hJjUejXPnziU6D/nMfUHm8xYgjQhYoUIFXya7NhEnMp8cJFHCj0c9pEwREAEREAEREAEREIEDJ5CjEqSvuqge77nnHsvEVXHAgAG+AtoVAREQAREQAREQARHIhQSSKUFWrlyZ2D1EBUeh2LFjR9awzoWAVCUREAEREAEREAEREAEfgWRKkMynnjRpkq9C2hUBERABERABERABEcjlBLJ3LnYub7yqJwIiIAIiIAIiIAIikAkCkiAzAU2niIAIiIAIiIAIiEBaE5AEmdaPX40XAREQAREQAREQgUwQyONWnQ4+uVatWtu2bWMhweBiKXD0ww8/dOviEMOSuJLEsEyBdqkJOUmAFdtZYCkn76h7hZ3Arl27WAehRIkSYW+I6p8UAnp/koI9ZW66devWiy+++Nlnn02oRfHOpDnrrLPOPPPMiy66KKGrh7EwUvKCBQus5gMHDmSBxObNm4exIapzEgmwhvtbb72VxAro1qEjwDpbbCyUELqaq8K5gYDen9zwFMJbh6FDhyLtJFr/eCVIlq4mZOPhhx+e6A3CWN41k+W2CxYs6HbD2BbVOSkEWM9Tr01SyOumIiACIiACiRJA2kHMS/Qs+UEmSkzlRUAEREAEREAERCDdCUiCTPc3QO0XAREQAREQAREQgUQJSIJMlJjKi4AIiIAIiIAIiEC6E5AEme5vgNovAiIgAiIgAiIgAokSSNhxMtEbBJdn+thLL730yiuvEEOnWrVqN95444UXXhhwyvbt2++///433njjt99+Y3p4586dzznnnIDyOvTDDz/cfffdXg44zB5zzDHM2z/jjDO8+UqLgAiIQJwEfvzxx549e0YtfOutt55++ukzZsyYOnVqZAHiowVEDPn555/nz5//5ptvLl++vEiRIvXq1aOTZxJn5HXiz9m7d+/XX39drly5+E/xlSQ4V4UKFXyZ2hUBEUiyBDlkyJC77rrrlFNOIVTQ66+/PmvWrBdffLFFixaxHsxVV101Z84c5pzTHZjoifSJMBSrvPKRuSdOnHjUUUcVL17caHz11Ve//vrr8OHDn3zyySuuuEKIREAERCBRAoQHHjduHOIgfYvv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)

### RoCE

`InfiniBand` 架构获得了极好的性能，但是其不仅要求在服务器上安装专门的 `InfiniBand` 网卡，还需要专门的交换机硬件，成本十分昂贵。而在企业界大量部署的是以太网络，为了复用现有的以太网，同时获得 `InfiniBand` 强大的性能，`IBTA` 组织推出了 `RoCE`（`RDMA over Converged Ethernet`）。`RoCE` 支持在以太网上承载 `IB` 协议，实现 `RDMA over Ethernet`，这样一来，仅需要在服务器上安装支持 `RoCE` 的网卡，而在交换机和路由器仍然使用标准的以太网基础设施。网络侧需要支持**无损以太网络**，这是由于 `IB` 的丢包处理机制中，任意一个报文的丢失都会造成大量的重传，严重影响数据传输性能。

`RoCE` 与 `InfiniBand` 技术有相同的软件应用层及传输控制层，仅网络层及以太网链路层存在差异，如下图所示：

![alt text](https://johng.cn/assets/images/image-22-1d297c874e1a0bdf104be594f0a2a482.png)

`RoCE`协议分为两个版本：

- **RoCE v1协议：** 基于以太网承载 `RDMA`，只能部署于二层网络，它的报文结构是在原有的 `IB` 架构的报文上增加二层以太网的报文头，通过 `Ethertype` `0x8915`标识 `RoCE` 报文。
- **RoCE v2协议：** 基于 `UDP/IP` 协议承载 `RDMA`，可部署于三层网络，它的报文结构是在原有的 `IB` 架构的报文上增加 `UDP` 头、`IP` 头和二层以太网报文头，通过 `UDP` 目的端口号 `4791` 标 识 `RoCE` 报文。`RoCE v2` 支持基于源端口号 `hash`，采用 `ECMP` 实现负载分担，提高了网络的利用率。

### iWARP

`iWARP` 从以下几个方面降低了主机侧网络负载：

- `TCP/IP` 处理流程从 `CPU` 卸载到 `RDMA` 网卡处理，降低了 `CPU` 负载。
- 消除内存拷贝：应用程序可以直接将数据传输到对端应用程序内存中，显著降低 `CPU` 负载。
- 减少应用程序上、下文切换：应用程序可以绕过操作系统，直接在用户空间对 `RDMA` 网卡下发命令，降低了开销，显著降低了应用程序上、下文切换造成的延迟。

由于 `TCP` 协议能够提供流量控制和拥塞管理，因此 `iWARP` 不需要以太网支持无损传输，仅通过普通以太网交换机和 `iWARP` 网卡即可实现，因此能够在广域网上应用，具有较好的扩展性。