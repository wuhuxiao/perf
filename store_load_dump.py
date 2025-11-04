import cupy as cp
import numpy as np
import time

NUM_LAYER = 16

BLOCK_SIZE = 128
NUM_BLOCK = 128
NUM_HEAD = 8
HEAD_SIZE = 128

# 64MB
HIDDEN_SIZE = NUM_BLOCK * BLOCK_SIZE * NUM_HEAD * HEAD_SIZE # 模拟每层KV大小（更大一些）

# 是layerwise 还是 blockwise
# TRANSFER_SIZE = NUM_BLOCK * BLOCK_SIZE * NUM_HEAD * HEAD_SIZE * np.float32().nbytes # 模拟每次load 和dump KV大小
TRANSFER_SIZE = BLOCK_SIZE * NUM_HEAD * HEAD_SIZE * np.float32().nbytes # 模拟每次load 和dump KV大小
# 加大 compute、load、dump 的耗时
LOAD_ITER = 1
COMPUTE_ITER = 2
DUMP_ITER = 1


# 三个 stream: load / compute / dump
stream_load = cp.cuda.Stream(non_blocking=True)
stream_compute = cp.cuda.Stream(non_blocking=True)
stream_dump = cp.cuda.Stream(non_blocking=True)

# 每层事件
events_load_done = [cp.cuda.Event() for _ in range(NUM_LAYER)]
events_compute_done = [cp.cuda.Event() for _ in range(NUM_LAYER)]
events_dump_done = [cp.cuda.Event() for _ in range(NUM_LAYER)]

# ======== 模拟 pinned memory（页锁定内存）========
host_kv_buffers = [
    cp.cuda.alloc_pinned_memory(HIDDEN_SIZE * np.float32().nbytes)
    for _ in range(NUM_LAYER)
]
device_kv_buffers = [cp.empty((BLOCK_SIZE * NUM_BLOCK, NUM_HEAD * HEAD_SIZE), dtype=cp.float32) for _ in range(NUM_LAYER)]
up_weights = cp.empty((NUM_HEAD * HEAD_SIZE, 2 * NUM_HEAD * HEAD_SIZE), dtype=cp.float32)
down_weights = cp.empty((2 * NUM_HEAD * HEAD_SIZE, NUM_HEAD * HEAD_SIZE), dtype=cp.float32)

time.sleep(0.5)

# ==== 模拟核函数开销 ====
def heavy_op(x):
    # 模拟一次 MLP 层： matmul -> 激活 -> matmul
    y = cp.matmul(x, up_weights)
    y = cp.tanh(y)
    y = cp.matmul(y, down_weights)
    return x

def gpu_compute(tensor):
    """模拟GPU前向计算 (多次 kernel 调用，消耗时间)"""
    for _ in range(COMPUTE_ITER):
        tensor = heavy_op(tensor)
    return tensor


def load_kv(layer_id):
    """从 host 异步拷贝到 device"""
    with stream_load:
        h_ptr = host_kv_buffers[layer_id]
        d_ptr = device_kv_buffers[layer_id]
        host_ptr = h_ptr.ptr
        device_ptr = d_ptr.data.ptr
        for _ in range(LOAD_ITER):
            for _ in range(d_ptr.nbytes//TRANSFER_SIZE):
                # 模拟异步 H2D 拷贝
                cp.cuda.runtime.memcpyAsync(
                    device_ptr, host_ptr, TRANSFER_SIZE,
                    cp.cuda.runtime.memcpyHostToDevice,
                    stream_load.ptr
                )
                device_ptr += TRANSFER_SIZE
                host_ptr += TRANSFER_SIZE

        events_load_done[layer_id].record(stream_load)
        print(f"[LOAD] layer {layer_id} enqueued (H2D + load op)")


def compute_layer(layer_id):
    """在 GPU 上计算，等待 load 完成"""
    with stream_compute:
        stream_compute.wait_event(events_load_done[layer_id])
        gpu_compute(device_kv_buffers[layer_id])
        events_compute_done[layer_id].record(stream_compute)
        print(f"[COMPUTE] layer {layer_id} done")


def dump_kv(layer_id):
    """异步将 device 拷贝回 host"""
    with stream_dump:
        stream_dump.wait_event(events_compute_done[layer_id])
        h_ptr = host_kv_buffers[layer_id]
        d_ptr = device_kv_buffers[layer_id]
        host_ptr = h_ptr.ptr
        device_ptr = d_ptr.data.ptr
        for _ in range(DUMP_ITER):
            for _ in range(d_ptr.nbytes//TRANSFER_SIZE):
                # 模拟 D2H 拷贝
                cp.cuda.runtime.memcpyAsync(
                    h_ptr.ptr, device_ptr, TRANSFER_SIZE,
                    cp.cuda.runtime.memcpyDeviceToHost,
                    stream_dump.ptr
                )
                device_ptr += TRANSFER_SIZE
                host_ptr += TRANSFER_SIZE
        events_dump_done[layer_id].record(stream_dump)
        print(f"[DUMP] layer {layer_id} enqueued (D2H + dump op)")


def Worker(num_layer):
    start = time.time()

    # Pipeline: load[layer+1], compute[layer], dump[layer-1]
    for i in range(num_layer + 2):
        if i < num_layer:
            load_kv(i)
        if 0 <= i - 1 < num_layer:
            compute_layer(i - 1)
        if 0 <= i - 2 < num_layer:
            dump_kv(i - 2)

    stream_load.synchronize()
    stream_compute.synchronize()
    stream_dump.synchronize()

    print(f"\n✅ All done, total time = {time.time() - start:.3f}s")


if __name__ == "__main__":

    Worker(2)
    time.sleep(0.5)

    Worker(NUM_LAYER)
