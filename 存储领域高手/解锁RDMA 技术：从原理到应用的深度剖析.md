# 解锁RDMA 技术：从原理到应用的深度剖析

在当今这个数据爆炸的时代，数据的高效传输和处理成为了各个领域的关键需求。无论是大规模的科学计算、海量数据的存储与分析，还是实时性要求极高的金融交易和人工智能训练，传统的网络通信技术在面对这些挑战时，都显得有些力不从心。而今天，我们要一起探索的 [RDMA](https://zhida.zhihu.com/search?content_id=251946061&content_type=Article&match_order=1&q=RDMA&zhida_source=entity) 技术，就像是一把神奇的钥匙，有望为我们打开高效网络通信的新大门，让数据传输变得更加快速、高效和低延迟

## 一、RDMA简介

### 1.1RDMA 是何方神圣？

RDMA，全称远程直接数据存取（Remote Direct Memory Access），是一种创新性的网络通信技术。在传统网络通信模式下，数据传输往往需要经过操作系统及多层软件协议栈的处理，这会导致大量的 CPU 资源被占用，数据传输延迟较高。而 RDMA 技术的出现，旨在解决这些问题，它能够让计算机直接访问远程计算机的内存，而无需在本地和远程计算机之间进行繁琐的数据复制，从而显著降低数据传输的延迟，提高数据处理效率，这使得它在现代网络通信中占据着至关重要的地位，尤其在对网络性能要求极高的领域，如高性能计算（HPC）、数据中心、云计算等，发挥着不可或缺的作用。

### 1.2技术背景

**⑴传统网络通信的困境**

在深入探讨 RDMA 技术之前，我们先来了解一下传统网络通信存在的问题。以常见的 TCP/IP 通信模式为例，当数据在网络中传输时，需要经过操作系统内核以及多层软件协议栈的处理。这一过程会涉及到多次的数据拷贝和上下文切换，从而导致较高的延迟和大量的 CPU 资源消耗。

具体来说，当发送端应用程序要发送数据时，数据首先会从用户空间的缓冲区拷贝到内核空间的缓冲区，然后经过协议栈的封装，再通过网卡发送出去。在接收端，数据则需要从网卡接收，经过协议栈的解封装，再从内核空间拷贝到用户空间的缓冲区，以供应用程序使用。这些数据拷贝和上下文切换操作不仅占用了宝贵的 CPU 时间，还使得数据传输的延迟难以降低，尤其是在处理大量小数据块的传输或者对实时性要求较高的应用场景时，传统网络通信的这些问题就显得更加突出，严重影响了系统的整体性能和响应速度。传统的 TCP/IP 网络通信，数据需要通过用户空间发送到远程机器的用户空间，在这个过程中需要经历若干次内存拷贝：

<img src="https://pica.zhimg.com/v2-57abe978fe46f6952215c38e8e800b68_1440w.jpg" alt="img" style="zoom: 67%;" />

<img src="https://picx.zhimg.com/v2-ac906f91f6f94ef696757b1c3412ace9_1440w.jpg" alt="img" style="zoom: 80%;" />

如上图，在传统模式下，两台服务器上的应用之间传输数据，过程是这样的：

- 首先要把数据从应用缓存拷贝到Kernel中的TCP协议栈缓存；
- 然后再拷贝到驱动层；
- 最后拷贝到网卡缓存。

多次内存拷贝需要CPU多次介入，导致处理延时大，达到数十微秒。同时整个过程中CPU过多参与，大量消耗CPU性能，影响正常的数据计算。

**⑵TCP/IP存在的问题**

传统TCP/IP通信存在的主要问题就是I/O瓶颈问题。在高速网络环境下与网络I/O相关的主机处理的高开销（数据移动操作和复制操作）限制了机器之间的传输带宽。

具体来说，传统的TCP/IP网络通信是通过内核发送消息。通过内核来传输消息这种机制会导致性能低和灵活性差。

- 性能低的主要原因是：由于网络通信通过`内核传递`，需要在内核中频繁进行协议封装和解封操作，造成很大的数据移动和数据复制开销。
- 灵活性差的原因是：是因为网络通信协议在`内核中进行处理`，这种方式很难支持新的网络协议和新的消息通信协议以及发送和接收接口。

## 二、RDMA 技术的核心原理

RDMA （绕过CPU，数据直接‘传’到对端内存）为了消除传统网络通信带给计算任务的瓶颈，我们希望更快和更轻量级的网络通信，由此提出了RDMA技术。RDMA利用 Kernel Bypass 和 Zero Copy技术提供了低延迟的特性，同时减少了CPU占用，减少了内存带宽瓶颈，提供了很高的带宽利用率。RDMA提供了给基于 IO 的通道，这种通道允许一个应用程序通过RDMA设备对远程的虚拟内存进行直接的读写。

**RDMA 技术有以下几个特点：**

CPU Offload：无需CPU干预，应用程序可以访问远程主机内存而不消耗远程主机中的任何CPU。远程主机内存能够被读取而不需要远程主机上的进程（或CPU)参与。远程主机的CPU的缓存(cache)不会被访问的内存内容所填充

Kernel Bypass：RDMA 提供一个专有的 Verbs interface 而不是传统的TCP/IP Socket interface。应用程序可以直接在用户态执行数据传输，不需要在内核态与用户态之间做上下文切换

Zero Copy：每个应用程序都能直接访问集群中的设备的虚拟内存，这意味着应用程序能够直接执行数据传输，在不涉及到网络软件栈的情况下，数据能够被直接发送到缓冲区或者能够直接从缓冲区里接收，而不需要被复制到网络层。

下面是 RDMA 整体框架架构图，从图中可以看出，RDMA 提供了一系列 Verbs 接口，可在应用程序用户空间，操作RDMA硬件。RDMA绕过内核直接从用户空间访问RDMA 网卡。RNIC(RDMA 网卡，RNIC（NIC=Network Interface Card ，网络接口卡、网卡，RNIC即 RDMA Network Interface Card）中包括 Cached Page Table Entry，用来将虚拟页面映射到相应的物理页面。

<img src="https://pic4.zhimg.com/v2-18e8039980f558c8ac42e7c41d86cbd9_1440w.jpg" alt="img" style="zoom: 67%;" />

### 2.1直接内存访问机制

RDMA 的核心在于其直接内存访问机制。在传统的网络通信中，数据传输需要 CPU 的深度参与，例如数据从应用程序缓冲区拷贝到内核缓冲区，再通过网络协议栈进行封装和传输，接收端则需要逆向操作，将数据从内核缓冲区拷贝到应用程序缓冲区，整个过程涉及多次数据拷贝和 CPU 上下文切换，效率较低。而 RDMA 允许计算机直接存取其他计算机的内存，绕过了处理器的繁琐处理过程，数据传输的大部分工作由硬件来执行，直接在远程系统的内存之间进行读写操作，极大地提高了数据传输的效率和速度，减少了 CPU 的负担，使得系统能够将更多的资源用于实际的数据处理任务，从而提升整体性能。

### 2.2零拷贝与内核旁路

```scss
磁盘 (disk)
   ↓  [DMA：设备→内核页缓存 page cache]
内核页缓存 (kernel page cache)
   ↓  [CPU memcpy：copy_to_user]
用户缓冲区 (user buffer)
   ↓  [CPU memcpy：copy_from_user]
socket 缓冲区 (kernel socket buffer)
   ↓  [DMA：网卡读内存]
网卡 (NIC)
```

零拷贝技术是 RDMA 的另一大关键特性。在传统通信模式下，数据在传输过程中需要在不同的内存区域之间进行多次拷贝，例如从用户空间拷贝到内核空间，再从内核空间拷贝到网络设备缓冲区等，这些拷贝操作不仅消耗 CPU 资源，还会增加数据传输的延迟。而 RDMA 实现了零拷贝，使得数据能够直接在应用程序的缓冲区与网络之间进行传输，无需中间的拷贝环节，大大减少了数据传输的开销和延迟。

**内核旁路**也是 RDMA 提升性能的重要手段。在传统网络通信中，应用程序与网络设备之间的通信需要经过操作系统内核的干预，这会导致上下文切换和系统调用的开销。而 RDMA 允许应用程序在用户态直接与网卡进行交互，避免了内核态与用户态之间的上下文切换，进一步降低了 CPU 的负担，提高了数据传输的效率和响应速度。例如，在一些对实时性要求极高的金融交易系统中，RDMA 的零拷贝和内核旁路技术能够确保交易数据的快速传输和处理，减少交易延迟，提高交易效率。

### 2.3RDMA通信协议

目前，有三种支持RDMA的通信技术：

1. [InfiniBand](https://zhida.zhihu.com/search?content_id=251946061&content_type=Article&match_order=1&q=InfiniBand&zhida_source=entity)(IB): 基于 InfiniBand 架构的 RDMA 技术，需要专用的 IB 网卡和 IB 交换机。从性能上，很明显Infiniband网络最好，但网卡和交换机是价格也很高。
2. [RoCE](https://zhida.zhihu.com/search?content_id=251946061&content_type=Article&match_order=1&q=RoCE&zhida_source=entity)：即RDMA over Ethernet(RoCE), 基于以太网的 RDMA 技术，也是由 IBTA 提出。RoCE支持在标准以太网基础设施上使用RDMA技术，但是需要交换机支持无损以太网传输，只不过网卡必须是支持RoCE的特殊的NIC。
3. [iWARP](https://zhida.zhihu.com/search?content_id=251946061&content_type=Article&match_order=1&q=iWARP&zhida_source=entity)：Internet Wide Area RDMA Protocal，基于 TCP/IP 协议的 RDMA 技术(在现有TCP/IP协议栈基础上实现RDMA技术,在TCP协议上增加一层DDP)，由 IETF 标 准定义。iWARP 支持在标准以太网基础设施上使用 RDMA 技术，而不需要交换机支持无损以太网传输，但服务器需要使用支持iWARP 的网卡。与此同时，受 TCP 影响，性能稍差。

这三种技术都可以使用同一套API来使用，但它们有着不同的物理层和链路层；需要注意的是，上述几种协议都需要专门的硬件（网卡）支持。

**⑴InfiniBand**

InfiniBand（IB）是一种服务器和存储器的互联技术，它具有高速、低延迟、低CPU负载、高效率和可扩展的特性。InfiniBand的关键特性之一是它天然地支持远程直接内存访问（RDMA）。InfiniBand能够让服务器与服务器之间、服务器与存储设备之间的数据传输不需要主机CPU的参与。
InfiniBand使用I/O通道进行数据传输，每个I/O通道提供虚拟的NIC或HCA语义。InfiniBand提供了多种技术方案，每个端口的速度可以有10GB/s、40GB/s、56GB/s、100GB/s，截止目前已经达到了200GB/s。InfiniBand使用同轴电缆和光纤进行连接。

**⑵RoCE**

RDMA首先从InfiniBand规范和产品里引入工业界，但是目前在企业界部署着大量基于以太网的产品，因此IBTA规范组织又定义了一套字符规范，使得RDMA不仅可以在infiniBand网络上运行，同时也可以在以太网上运行。
RoCE是基于以太网（Ethernet）的RDMA技术标准，它也是由IBTA组织指定的。RoCE为以太网提供了RDMA语义，并不需要复杂低效的TCP传输（IWARP需要）。
RoCE是现在最有效的以太网低延迟方案。它消耗很少的CPU负载，在数据中心桥接以太网中利用优先流控制（PFC）来达到网络的无损连接。
RoCE 有两个版本，RoCE v1是一种链路层协议，允许在同一个广播域下的任意两台主机直接访问。RoCE v2是一种Internet层协议，即可以实现路由功能。虽然RoCE协议这些好处都是基于融合以太网的特性，但是RoCE协议也可以使用在传统以太网网络或者非融合以太网络中。

**⑶iWARP**

这是一种基于 TCP 的 RDMA 网络协议，它利用了 TCP 的可靠性来实现远程内存访问，能够在现有的 TCP/IP 网络基础上部署 RDMA 技术，具有良好的兼容性和广泛的适用性。然而，由于 TCP 协议本身的一些特性，如在大型组网情况下，大量的 TCP 连接会占用较多的内存资源，对系统规格要求相对较高。不过，在一些对网络兼容性要求较高，且数据传输量相对较小、对延迟不太敏感的场景中，iWARP 协议仍然具有一定的优势，例如一些企业内部的小型网络环境或者对成本控制较为严格的场景。

## 三、RDMA编程详解

### 3.1传输操作

RDMA有两种基本操作，包括[Memory verbs](https://zhida.zhihu.com/search?content_id=251946061&content_type=Article&match_order=1&q=Memory+verbs&zhida_source=entity)和Messaging verbs。

**⑴Memory verbs：包括read、write和atomic操作。**

RDMA Read：从远程主机读取部分内存。调用者指定远程虚拟地址，像本地内存地址一样用来拷贝。在执行 RDMA 读操作之前，远程主机必须提供适当的权限来访问它的内存。一旦权限设置完成， RDMA 读操作就可以在对远程主机没有任何通知的条件下执行。不管是 RDMA 读还是 RDMA 写，远程主机都不会意识到操作正在执行 （除了权限和相关资源的准备操作）。

RDMA Write：与RDMA Read类似，只是数据写到远端主机中。RDMA写操作在执行时不通知远程主机。然而带即时数的RDMA写操作会将即时数通知给远程主机。

RDMA Atomic：包括原子取、原子加、原子比较和原子交换，属于RDMA原子操作的扩展。

**⑵Messaging verbs：包括send和receive操作。**

RDMA Send：发送操作允许你把数据发送到远程 QP 的接收队列里。接收端必须已经事先注册好了用来接收数据的缓冲 区。发送者无法控制数据在远程主机中的放置位置。可选择是否使用即时数，一个4位的即时数可以和数据缓冲一起被传送。这个即时数发送到接收端是作为接收的通知，不包含在数据缓冲之中。

RDMA Receive：这是与发送操作相对应的操作。接收主机被告知接收到数据缓冲，还可能附带一个即时数。接收端应用 程序负责接收缓冲区的维护和发布。

### 3.2传输模式

按照连接和可靠两个标准，可以划分出四种不同的传输模式：

- 可靠连接（RC）：一个QP只和另一个QP相连，消息通过一个QP的发送队列可靠地传输到另一个QP的接收队列。数据包按序交付，RC连接很类似于TCP连接。
- 不可靠连接（UC）：一个QP只和另一个QP相连，连接是不可靠的，所以数据包可能有丢失。传输层出错的消息不会进行重传，错误处理必须由高层的协议来进行。
- 不可靠数据报（UD）：一个 QP 可以和其它任意的 UD QP 进行数据传输和单包数据的接收。不保证按序性和交付性。交付的数据包可能被接收端丢弃。支持多播消息（一对多）。UD连接很类似于UDP连接。

### 3.3 RDMA的编程接口

RDMA 的编程接口主要包括 [Verbs API](https://zhida.zhihu.com/search?content_id=251946061&content_type=Article&match_order=1&q=Verbs+API&zhida_source=entity) 和 RDMA CM（Connection Manager）API 等，其中 Verbs API 提供了一套完整的 RDMA 操作函数，是实现 RDMA 功能的关键所在。

在使用 Verbs API 进行编程时，首先需要进行设备查询和初始化操作，例如使用ibv_get_device_list函数获取系统中的 RDMA 设备列表，然后选择合适的设备并通过ibv_open_device函数打开设备，获取设备上下文（ibv_context），这一步是后续所有操作的基础，只有正确地获取并初始化设备，才能进行后续的内存注册、队列对创建等操作。

内存注册是通过ibv_reg_mr函数来实现的，它将内存块固定（防止被交换出）并返回一个包含uint32_t键（key）的ibv_mr结构体指针，这个key允许远程访问注册的内存，同时该结构体还包含了内存区域的上下文（context）、地址（addr）、长度（length）等信息，这些信息在后续的数据传输操作中起着重要作用。

队列对（[Queue Pair](https://zhida.zhihu.com/search?content_id=251946061&content_type=Article&match_order=1&q=Queue+Pair&zhida_source=entity)）的创建和管理也是使用 Verbs API 的重要部分。首先，需要使用ibv_alloc_pd函数分配一个保护域（Protection Domain），然后通过ibv_create_cq函数创建完成队列（Completion Queue），并使用ibv_create_qp函数创建队列对，包括发送队列（Send Queue）和接收队列（Receive Queue）。在创建队列对时，需要配置一些参数，如队列的深度、服务级别等，以满足不同的应用需求。

数据发送和接收则是通过ibv_post_send和ibv_post_recv函数来完成的。对于发送操作，需要先构建一个ibv_send_wr结构体，设置好操作码（opcode）、工作队列元素列表（sg_list）、远程地址（remote_addr）、远程键（rkey）等参数，然后将其提交到发送队列中；接收操作类似，构建ibv_recv_wr结构体并提交到接收队列中。当操作完成后，完成队列中会生成相应的完成队列元素（Completion Queue Element，CQE），通过轮询完成队列可以获取操作的完成状态和结果信息。

以下是一个简单的伪代码示例，展示了如何使用 Verbs API 进行 RDMA 数据传输：

```cpp
#include <infiniband/verbs.h>

// 全局变量定义
struct ibv_context *ctx;
struct ibv_pd *pd;
struct ibv_mr *mr;
struct ibv_qp *qp;
struct ibv_cq *cq;

// 初始化RDMA资源
void init_rdma() {
    // 查询RDMA设备
    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    if (!dev_list) {
        perror("Failed to get RDMA device list");
        exit(1);
    }
    ctx = ibv_open_device(dev_list[0]);
    if (!ctx) {
        perror("Failed to open RDMA device");
        exit(1);
    }
    // 创建保护域
    pd = ibv_alloc_pd(ctx);
    if (!pd) {
        perror("Failed to allocate protection domain");
        exit(1);
    }
    // 注册内存区域
    char *buf = malloc(1024);
    mr = ibv_reg_mr(pd, buf, 1024, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!mr) {
        perror("Failed to register memory region");
        exit(1);
    }
    // 创建完成队列
    cq = ibv_create_cq(ctx, 10, NULL, NULL, 0);
    if (!cq) {
        perror("Failed to create completion queue");
        exit(1);
    }
    // 创建队列对
    struct ibv_qp_init_attr qp_attr = {
      .send_cq = cq,
      .recv_cq = cq,
      .cap = {
          .max_send_wr = 10,
          .max_recv_wr = 10,
          .max_send_sge = 1,
          .max_recv_sge = 1
       },
      .qp_type = IBV_QPT_RC
    };
    qp = ibv_create_qp(pd, &qp_attr);
    if (!qp) {
        perror("Failed to create queue pair");
        exit(1);
    }
}

// 发送数据
void send_data() {
    struct ibv_send_wr wr, *bad_wr;
    struct ibv_sge sge;
    memset(&wr, 0, sizeof(wr));
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uint64_t)mr->addr;
    sge.length = 1024;
    sge.lkey = mr->lkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    // 设置远程地址和键（假设已获取）
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = remote_rkey;
    if (ibv_post_send(qp, &wr, &bad_wr)) {
        perror("Failed to post send");
        exit(1);
    }
}

// 接收数据
void receive_data() {
    struct ibv_recv_wr wr, *bad_wr;
    struct ibv_sge sge;
    memset(&wr, 0, sizeof(wr));
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uint64_t)mr->addr;
    sge.length = 1024;
    sge.lkey = mr->lkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    if (ibv_post_recv(qp, &wr, &bad_wr)) {
        perror("Failed to post receive");
        exit(1);
    }
}

// 轮询完成队列获取完成事件
void poll_cq() {
    struct ibv_wc wc;
    while (ibv_poll_cq(cq, 1, &wc)) {
        if (wc.status == IBV_WC_SUCCESS) {
            // 处理完成事件
            printf("RDMA operation completed successfully\n");
        } else {
            perror("RDMA operation failed");
        }
    }
}

// 释放RDMA资源
void cleanup() {
    ibv_dereg_mr(mr);
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
}
```

在上述代码中，首先通过init_rdma函数完成了 RDMA 设备的查询、打开，以及保护域、内存区域、完成队列和队列对的创建等初始化操作。然后，send_data函数用于向远程节点发送数据，receive_data函数用于接收远程节点发送的数据，poll_cq函数则通过轮询完成队列来获取操作的完成状态信息。最后，cleanup函数用于释放之前分配的 RDMA 资源，确保程序的正确退出和资源的有效回收。

通过这些函数的组合使用，开发者可以在应用程序中充分利用 RDMA 技术的优势，实现高效的远程数据传输和处理，提高系统的整体性能和响应速度。

### 3.4内存注册与队列对

内存注册是 RDMA 操作的重要前提。由于 RDMA 硬件对用于数据传输的内存有特殊要求，在数据传输过程中，应用程序不能修改数据所在的内存，操作系统也不能对其进行 page out 操作，即物理地址和虚拟地址的映射必须固定不变。而且，无论是 DMA 还是 RDMA 都要求物理地址连续，这是由 DMA 引擎所决定的。

内存注册的过程是通过创建两个键（local和remote）指向需要操作的内存区域来实现的。注册的键是数据传输请求的一部分，注册一个Memory Region（内存区域）后，该区域会具有一些属性，如上下文（context）、地址（addr）、长度（length）、本地键（lkey）和远程键（rkey）等。只有将要操作的内存注册到 RDMA 内存区域中，这块内存才能交给 RDMA 保护域来操作，之后便可以对该内存进行 RDMA 操作，只要保证接收方的缓冲区接收长度大于等于发送方的缓冲区长度即可。

队列对（Queue Pair，QP）在 RDMA 通信中起着关键的调度作用。它由发送队列（Send Queue，SQ）和接收队列（Receive Queue，RQ）组成，任何通信过程都要有收发两端，QP 就是这两个队列的组合。发送队列专门用来存放发送任务，接收队列专门用来存放接收任务。在一次 SEND - RECV 流程中，发送端需要把表示一次发送任务的工作队列元素（Work Queue Element，WQE）放到发送队列中，接收端软件则需要给硬件下发一个表示接收任务的 WQE，这样硬件才知道收到数据后放到内存中的哪个位置。

完成队列（Completion Queue，CQ）与工作队列紧密配合。CQ 中的元素是完成队列元素（Completion Queue Element，CQE），如果说 WQE 是软件下发给硬件的 “任务书”，那么 CQE 就是硬件完成任务之后返回给软件的 “任务报告”。CQE 中描述了某个任务是被正确无误地执行，还是遇到了错误，如果遇到错误，还会说明错误的原因。每个 CQE 都包含某个 WQE 的完成信息，当发送端或接收端的硬件完成任务后，会生成 CQE 并放置到 CQ 中，上层应用通过轮询 CQ 来获取任务的完成信息，从而得知数据传输的状态和结果，以便进行后续的处理操作。

为了更清晰地展示数据传输过程中各组件的交互关系，以下是一个简单的序列图：

```html
@startuml
actor 发送端应用程序 as SenderApp
actor 接收端应用程序 as ReceiverApp
participant 发送端硬件 as SenderHW
participant 接收端硬件 as ReceiverHW
participant 发送队列 as SendQueue
participant 接收队列 as ReceiveQueue
participant 完成队列 as CompletionQueue

SenderApp -> SendQueue: 提交发送任务（WQE）
SendQueue -> SenderHW: 通知有任务
SenderHW -> 发送端内存: 读取数据
SenderHW -> ReceiverHW: 发送数据
ReceiverHW -> 接收端内存: 写入数据
ReceiverHW -> ReceiveQueue: 放置接收完成信息
ReceiverApp -> ReceiveQueue: 轮询接收完成情况
SenderHW -> CompletionQueue: 放置发送完成信息（CQE）
SenderApp -> CompletionQueue: 轮询发送完成情况
@enduml
```

在这个序列图中，发送端应用程序将发送任务的 WQE 提交到发送队列，发送队列通知发送端硬件有任务待处理，发送端硬件从内存读取数据并发送给接收端硬件，接收端硬件将数据写入内存后，在接收队列放置接收完成信息，同时在完成队列放置发送完成信息，发送端和接收端应用程序通过轮询相应队列来获取任务完成情况，从而实现了 RDMA 数据的可靠传输和任务的协调调度。这种通过队列对和完成队列的协同工作机制，使得 RDMA 能够高效地进行数据传输，充分发挥其低延迟、高带宽的优势，满足各种对网络性能要求苛刻的应用场景的需求。

## 四、RDMA 的应用领域

### 4.1数据中心网络的变革

在大规模数据中心中，RDMA 技术正发挥着至关重要的作用，深刻改变着服务器间的通信模式。传统的 TCP/IP 通信方式在数据中心的海量数据传输场景下，往往显得力不从心，存在着高延迟、低带宽利用率以及 CPU 资源消耗大等问题。而 RDMA 技术的出现，犹如一场及时雨，为这些问题提供了有效的解决方案。

以云存储服务为例，当用户请求从云端下载或上传大量数据时，使用 RDMA 技术可以显著加速数据的传输过程。在一个拥有数千台服务器的大型数据中心中，采用 RDMA 技术的云存储系统，其数据传输速度相比传统方式可提升数倍甚至更高。这意味着用户能够更快地获取所需的数据，大大提高了用户体验。同时，对于数据中心的运营者来说，这也意味着能够在相同的时间内处理更多的用户请求，从而提高了服务的效率和竞争力。

在分布式数据库系统中，RDMA 同样展现出了强大的优势。在进行大规模数据的分布式查询和事务处理时，RDMA 技术能够实现服务器之间的高速数据交互，大大降低了数据传输的延迟。这使得数据库系统能够更快速地响应复杂的查询请求，提高了事务处理的效率和并发能力。例如，在一些大型电商平台的订单处理系统中，RDMA 技术的应用使得订单的查询和处理速度得到了显著提升，能够在高并发的情况下依然保持稳定的性能，确保了平台的高效运行。

此外，在应对数据中心内部的海量数据迁移和备份场景时，RDMA 技术也能够发挥其高带宽、低延迟的优势，大大缩短数据迁移和备份的时间窗口，减少对业务的影响，提高数据中心的整体可靠性和可用性。

### 4.2高性能计算的得力助手

在高性能计算（HPC）领域，RDMA 技术已经成为不可或缺的关键技术之一，为科学研究和工程计算等领域的发展提供了强大的动力。

在气象模拟领域，科学家们需要处理海量的气象数据，并进行复杂的数值计算，以预测天气变化和气候趋势。传统的网络通信方式往往无法满足这种大规模数据处理和高速计算的需求，而 RDMA 技术的应用则带来了显著的改变。通过 RDMA，计算节点之间能够实现高速的数据共享和交互，大大缩短了数据传输的时间，从而加速了整个气象模拟的计算进程。例如，在一个全球气候模拟项目中，使用 RDMA 技术后，数据传输的延迟降低了数倍，使得模拟计算的效率得到了大幅提升，原本需要数周甚至数月才能完成的模拟任务，现在能够在更短的时间内得到结果，为气象研究提供了更及时、准确的支持。

在基因测序领域，RDMA 同样发挥着重要作用。基因测序会产生海量的基因数据，对这些数据的分析和处理需要强大的计算能力和高效的数据传输能力。RDMA 技术使得基因测序设备与计算节点之间能够快速传输数据，减少了数据传输的瓶颈，提高了测序数据的处理速度。这不仅有助于加速科研项目的进展，例如在癌症基因研究、遗传疾病诊断等方面，能够更快速地分析基因数据，为疾病的诊断和治疗提供更有力的依据，而且也为基因技术在医学、农业等领域的广泛应用奠定了基础。

许多科研机构和企业在其高性能计算集群中广泛应用 RDMA 技术，取得了显著的性能提升和科研成果加速的效果。例如，某知名科研机构在其超级计算机上采用 RDMA 技术后，在一系列复杂的科学计算任务中，计算效率提高了 30% 以上，使得原本需要耗费大量时间和资源的科研项目能够更快地取得突破，推动了相关学科的发展和技术的进步。

### 4.3分布式存储系统的优化

在分布式存储系统中，RDMA 技术的应用为数据的高效存储和快速访问提供了有力保障。

传统的分布式存储架构在数据读写过程中，往往需要经过多个中间节点的处理和数据复制，这会导致较高的延迟和较低的读写性能。而 RDMA 技术的出现，使得存储节点和计算节点之间能够直接进行快速的数据传输，绕过了中间节点的繁琐处理，大大减少了数据复制和中间节点处理的开销。

以 Ceph、GlusterFS 等分布式存储系统为例，当计算节点需要从存储节点读取大量数据时，RDMA 技术能够实现数据的直接内存访问，避免了传统方式下数据在多个层次之间的拷贝和协议栈的处理，从而显著提高了数据读取的速度。在实际的应用场景中，如大规模的视频存储和在线播放平台，使用 RDMA 技术的分布式存储系统能够更快地响应用户的播放请求，减少视频的缓冲时间，提升用户的观看体验。

在数据写入方面，RDMA 同样表现出色。例如，在一些对数据写入速度要求极高的大数据存储场景中，如实时数据采集和存储系统，RDMA 技术能够确保数据快速、准确地写入存储节点，提高了存储系统的写入性能和响应速度，保证了数据的及时性和完整性。

此外，RDMA 技术还能够提高分布式存储系统的可靠性和可用性。通过减少中间节点的故障点，以及快速的数据恢复和同步能力，RDMA 技术使得分布式存储系统在面对节点故障或网络异常时，能够更快速地进行数据的冗余备份和恢复，确保数据的安全性和可用性，为企业的关键业务数据提供了更可靠的存储解决方案。

## 五、RDMA通信过程

为了执行 RDMA 操作，首选需要建立与远程主机的连接和适当的认证。实现这些的机制是队列对（QP） 。与标准的 IP 协议栈类似，一个 QP 大概等同于一个接字（socket）。 QP 需要在连接两端进行初始化。 连接管理器（CM）用来在 QP 建立之前进行 QP 信息的交换。一旦一个 QP 建立起来， verbs API 就可以用来执行 RDMA 读/写和原子操作。与套接字的读/写类似的连续收/发操作也能执行。RDMA的操作过程大致如下：

- 当一个应用程执行度或者写请求时，不执行任何数据复制，再不需要任何内核参与的条件下，RDMA请求从用户空间中的应用发送到本地NIC网卡。
- NIC读取缓冲区的内容，并通过网络传输到远程NIC。
- 在网络上传输的RDMA信息包括虚拟地址、内存钥匙和数据本身。请求既可以完全在用户空间中处理，又或者在应用一直睡眠到请求完成时的情况下通过系统级中断处理。RDMA操作使应用可以从一个远程应用的内存中读取数据或向这个内存中写数据。
- 目标NIC确认内存钥匙，直接将数据写入应用缓存中，用于操作的远程虚拟内存地址包含在RDMA信息中。

在RDMA操作中，Read/Write是单边操作，秩序本地端明确信息的源和目的地址，远端应用不必感知此次通信，数据的读或写都通过RDMA在RNIC与应用Buffer之间完成，再由远端RNIC封装成消息返回到本地端。Send/Receive是双边操作，即必须要远端的应用感知参与才能完成收发，在实际中，Send/Receive多用于连接控制类报文，而数据报文是通过Read/Write来完成的。

### 5.1单向通信-读Read

![img](https://picx.zhimg.com/v2-27424f33f746df616eab80f8fad6bb61_1440w.jpg)

- 首先A、B建立连接，QP已经创建并初始化。
- 数据被存档在B的buffer，地址为VB，注意VB是提前注册到B的RNIC，并且它是一个Memory Region，并拿到返回的local key，相当于RDMA操作这块buffer的权限。
- B把数据地址VB，key封装到专用报文传送到A，这相当于B把数据buffer的操作权交给了A。同时B在他的WQ中注册一个WR，用于接收数据传输的A返回的状态。
- A在收到B发送过来的数据地址VB和R_key之后，RNIC会把它们连同本地存储数据的地址VA封装到Read请求中，将这个请求消息发送到B，这个过程A、B两端不需任何软件参与，就可以将B中的数据存储到A的VA虚拟地址。
- A在存储完成后，向B返回数据传输的状态信息。

### 5.2 单向通信-写Write

![img](https://pica.zhimg.com/v2-98e561f071117d8a4d55b15114b46780_1440w.jpg)

- 首先A、B建立连接，QP已经创建并初始化。
- 数据远端的目标存储空间buffer的地址为VB，注意VB是提前注册到B的RNIC，并且它是一个Memory Region，并拿到返回的local key，相当于RDMA操作这块buffer的权限。
- B把数据地址VB，key封装到专用报文传送到A，这相当于B把数据buffer的操作权交给了A。同时B在他的WQ中注册一个WR，用于接收数据传输的A返回的状态。
- A在收到B发送过来的数据地址VB和R_key之后，RNIC会把它们连同本地存储数据的地址VA封装到Write请求中，这个过程A、B两端不需任何软件参与，就可以将A中的数据存储到B的VB虚拟地址。
- A在发送数据完成后，向B返回数据传输的状态信息。

### 5.3双向通信-Send\Recv

![img](https://pic3.zhimg.com/v2-18c0f9b675a327a5423d479d2a6e04ec_1440w.jpg)

- 首先，A和B都要创建并初始化好各自的QP、CQ。
- A、B分别想自己的WQ中注册WQE，对于A来说，WQ=SQ，WQE描述指向一个等待被发送的数据；对于B，WQ=RQ，WQE描述指向一块用于存储数据的Buffer。
- A的RNIC异步调度轮到A的WQE，解析到这是一个Send消息，从Buffer中直接向B发送数据。数据流到达B的RNIC后，B的WQE被消耗，并把数据直接存储到WQE指向的存储位置。
- AB通信完成后，A的CQ中会产生一个完成消息CQE表示发送完成。同时，B的CQ中会产生一个完成消息CQE表示接收完成。每个WQ中的WQE的处理完成都会产生一个CQE。

## 六、全文总结

RDMA 技术作为现代网络通信领域的一项关键技术，以其独特的直接内存访问、零拷贝和内核旁路等特性，有效地解决了传统网络通信中的高延迟、低带宽利用率以及 CPU 资源消耗大等问题，为众多对网络性能要求苛刻的应用场景提供了高效、可靠的数据传输解决方案。从数据中心的大规模数据处理与存储，到高性能计算领域的科学研究和工程计算，再到分布式存储系统的优化，RDMA 技术都展现出了强大的优势和巨大的应用潜力，推动了各行业的技术进步和业务创新，成为推动现代信息技术发展的重要力量。

尽管目前 RDMA 技术在实际应用中还面临着网络环境稳定性要求高、硬件成本相对较高以及技术普及难度较大等挑战，但随着网络优化技术的不断发展、硬件成本的逐渐下降以及开源社区和行业标准的持续推动，这些问题正在逐步得到解决。展望未来，RDMA 技术有望在硬件性能提升、应用领域拓展以及与其他新兴技术融合等方面取得更多的突破和发展，为网络通信和计算领域带来更多的创新和机遇，持续推动各行业数字化转型和智能化升级，其在未来信息技术领域的重要性和影响力也将不断提升，值得我们持续关注和深入探索其更多的应用可能性。