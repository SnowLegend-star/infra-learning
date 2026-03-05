# Linux IO模型详解

![img](https://i-blog.csdnimg.cn/blog_migrate/2f6061d0ac2262549d7fcc8a489971d8.png)

【**spdk是基于NVME协议的，二者不能并列**】

| 类别             | 举例                       | 关键点             |
| ---------------- | -------------------------- | ------------------ |
| Kernel Native IO | POSIX IO、libaio、io_uring | 仍经内核 I/O 栈    |
| Sync IO          | read/write + epoll         | 同步 + 非阻塞      |
| Async IO         | io_uring/libaio            | 真异步             |
| Kernel Bypass IO | DPDK、SPDK                 | 用户态驱动直访硬件 |