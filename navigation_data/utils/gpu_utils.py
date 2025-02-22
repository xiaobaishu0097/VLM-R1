from typing import Union

import pynvml
from torch.multiprocessing import Manager


def get_gpu_with_most_free_memory(gpu_indices: list) -> Union[int, int]:
    # Initialize NVML to interact with NVIDIA drivers
    pynvml.nvmlInit()

    # Dictionary to hold free memory for each GPU
    gpu_free_memory = {}

    for gpu_index in gpu_indices:
        try:
            # Get the handle for the current GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            # Retrieve memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # Store the free memory of the current GPU
            gpu_free_memory[gpu_index] = mem_info.free
        except pynvml.NVMLError as error:
            print(f"Failed to get memory info for GPU {gpu_index}: {error}")
            gpu_free_memory[gpu_index] = -1  # Assign negative value in case of error

    # Shut down NVML
    pynvml.nvmlShutdown()

    # Find the GPU index with the maximum free memory
    max_free_memory_gpu = max(gpu_free_memory, key=gpu_free_memory.get)

    return max_free_memory_gpu, gpu_free_memory[max_free_memory_gpu]


def release_gpu(gpu_manager, gpu_id):
    gpu_manager[gpu_id].set()


def init_gpu_manager(gpu_ids):
    manager = Manager()
    gpu_manager = manager.dict()
    for gpu_id in gpu_ids:
        gpu_manager[gpu_id] = manager.Event()  # 创建一个事件对象并设置为未触发状态
        gpu_manager[gpu_id].set()  # 设置事件，表示GPU可用
    return gpu_manager


def request_gpu(gpu_manager):
    for gpu_id, event in gpu_manager.items():
        if event.is_set():
            event.clear()
            return gpu_id
    return None
