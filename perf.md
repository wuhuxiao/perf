# H20 performance baseline

Chip Name	GH100
SM Count	78
L2 Cache Size	60.00 MiB
Memory Bandwidth	3,746.51 GiB/s
Memory Size	95.00 GiB
Core Clock	1.98 GHz
Bus Location	0000:98:00.0
UUID	d6148c49-d87e-a63f-0dac-b83c3a228349
GSP firmware version	535.183.01
Video accelerator tracing	Not supported: GSP enabled

CUDA 的数据传输是通过 GPU 的 **DMA 引擎** 实现的：

| 方向 | 主控设备 | 数据源 | 数据目标 | 实际瓶颈 |
|------|-----------|---------|-----------|-----------|
| H2D  | GPU DMA 引擎主动发起 | 主机内存（pinned） | GPU 显存 | GPU 端 DMA 负责 |
| D2H  | GPU DMA 引擎主动发起 | GPU 显存 | 主机内存（pinned） | CPU 内存系统参与写入 |

区别在于：

- **H2D 时**，GPU 主动从主机内存中**读取数据**；
- **D2H 时**，GPU 主动往主机内存**写入数据**。

写入主机内存时会触发 CPU 内存控制器的 **缓存一致性（cache coherence）机制**，导致：

- CPU 需要标记对应页为无效；
- 写入路径需经过 write combining 缓冲；
- 顺序写入路径 latency 更高、带宽利用率更差。

## 测试环境和脚本
```bash
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true --sample=none --trace-fork-before-exec=true \ 
--cuda-graph-trace=node --force-overwrite=true --output=/home/externals/wangwenxin21/perf/base python store_load_dump.py
```

## 实测性能 

| 方向 | 数据包大小 64MB（128 * 128 * 8 * 128 * 4） | 带宽                    | 时延    |
|------|-------------------------------------|-----------------------|---------|
| H2D  | layer wise一次load 连续的128个KV BLOCK    | 51.34 GiB/s | 1216.875 μs | 
| D2H  | layer wise一次dump 连续的128个KV BLOCK    | 45.90 GiB/s | 1361.250 μs |


| 方向 | 数据包大小 0.5 MB（1 * 128 * 8 * 128 * 4） | 带宽                    | 时延    |
|------|-------------------------------------|-----------------------|---------|
| H2D  | layer wise 128次load 1个KV BLOCK      | 38.48 GiB/s | 12.764 μs | 
| D2H  | layer wise 128次dump 1个KV BLOCK      | 37.07 GiB/s | 13.252 μs |