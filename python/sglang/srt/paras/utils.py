import torch
import functools
import pynvml
import pickle
import numpy as np
import torch.distributed as dist
from typing import List

def print_nvml_mem_for_torch_device(device_id: int):
    """
    Print NVML GPU memory usage for a given torch.device
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()

    print(f"[NVML] {device_id}:")
    print(f"  Total: {mem.total/1024**3:.2f} GB")
    print(f"  Used:  {mem.used/1024**3:.2f} GB")
    print(f"  Free:  {mem.free/1024**3:.2f} GB")

def paras_memory_check(checkpoint: str = ""):
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"ParaS {checkpoint} Memory Allocated: {allocated:.2f} GB, "
          f"Reserved: {reserved:.2f} GB, "
          f"Max Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")

    print_nvml_mem_for_torch_device(torch.cuda.current_device())

def paras_func(func):
    """
    Decorator to ensure that the function is called with the `paras_configure_helper`
    """
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if hasattr(self, 'paras_configure_helper'):
            self.paras_configure_helper()
        return result
    return wrapper

def paras_profile_func(op_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "paras",
                    worker_name=f"{op_name}_rank{self.tp_rank}",
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            profiler.start()
            try:
                return func(self, *args, **kwargs)
            finally:
                profiler.stop()
        return wrapper
    return decorator

class ParaSWeightBuffer:
    def __init__(self):
        self.buffer: dict[int, list[torch.Tensor]] = {}

    def has(self, numel: int):
        return numel in self.buffer and len(self.buffer[numel]) > 0

    def get_buffer_like(self, tensor: torch.Tensor):
        numel = tensor.numel()
        if numel in self.buffer and len(self.buffer[numel]) > 0:
            return self.buffer[numel].pop()
        else:
            return torch.empty_like(tensor)
    
    def get_buffer(self, shape, dtype, device):
        numel = 1
        for dim in shape:
            numel *= dim
        if numel in self.buffer and len(self.buffer[numel]) > 0:
            # NOTE(shaoyuw): this assumes that all tensors have the same dtype and device
            tensor = self.buffer[numel].pop()
            return tensor.view(shape)
        else:
            return torch.empty(shape, dtype=dtype, device=device)

    def put(self, tensor: torch.Tensor):
        numel = tensor.numel()
        if numel not in self.buffer:
            self.buffer[numel] = []
        self.buffer[numel].append(tensor)

    def release_all(self):
        self.buffer.clear()
        torch.cuda.empty_cache()

paras_weight_buffer = ParaSWeightBuffer()