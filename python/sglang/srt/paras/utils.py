import torch
import sys
import functools
import pynvml
import pickle
import numpy as np
import torch.distributed as dist
from typing import List

from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, fields

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

def detect_tensor_in_class(
    obj,
    *,
    include_properties=True,
    recurse_containers=True,
    max_depth=5,
):
    """
    Yield (path, tensor) for ALL torch tensors reachable from obj.
    """
    seen = set()

    def walk(x, path, depth):
        if id(x) in seen:
            return
        seen.add(id(x))

        # 1. Direct tensor
        if torch.is_tensor(x):
            yield path, x
            return

        if depth >= max_depth:
            return

        # 2. Dataclass
        if is_dataclass(x):
            for f in fields(x):
                try:
                    v = getattr(x, f.name)
                except Exception:
                    continue
                yield from walk(v, f"{path}.{f.name}", depth + 1)
            return

        # 3. Mapping
        if isinstance(x, Mapping):
            for k, v in x.items():
                yield from walk(v, f"{path}[{k!r}]", depth + 1)
            return

        # 4. Sequence (but not str/bytes)
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
            for i, v in enumerate(x):
                yield from walk(v, f"{path}[{i}]", depth + 1)
            return

        # 5. Object attributes
        if hasattr(x, "__dict__"):
            for name, v in x.__dict__.items():
                yield from walk(v, f"{path}.{name}" if path else name, depth + 1)

        # 6. __slots__
        if hasattr(x, "__slots__"):
            for name in x.__slots__:
                try:
                    v = getattr(x, name)
                except Exception:
                    continue
                yield from walk(v, f"{path}.{name}" if path else name, depth + 1)

        # 7. Properties (optional, expensive)
        if include_properties:
            for name in dir(x):
                if name.startswith("_"):
                    continue
                try:
                    attr = getattr(type(x), name, None)
                    if isinstance(attr, property):
                        v = getattr(x, name)
                        yield from walk(v, f"{path}.{name}", depth + 1)
                except Exception:
                    pass

    yield from walk(obj, "", 0)
    
def print_class_tensor_member(obj):
    for path, tensor in detect_tensor_in_class(obj):
        print(f"{path}: tensor(shape={tuple(tensor.shape)}, "
              f"dtype={tensor.dtype}, device={tensor.device})")

def sizeof(obj, *, _seen=None):
    """
    Recursively compute memory usage of an object in bytes.
    Avoids double counting via object id memo.
    """
    if _seen is None:
        _seen = set()

    obj_id = id(obj)
    if obj_id in _seen:
        return 0
    _seen.add(obj_id)

    # Torch tensor: count underlying storage
    if torch.is_tensor(obj):
        # storage().nbytes() is the real payload
        return obj.storage().nbytes()

    size = sys.getsizeof(obj)

    # Dict
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            size += sizeof(k, _seen=_seen)
            size += sizeof(v, _seen=_seen)
        return size

    # List / tuple / set
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            size += sizeof(v, _seen=_seen)
        return size

    # Custom object
    if hasattr(obj, "__dict__"):
        size += sizeof(obj.__dict__, _seen=_seen)

    # __slots__
    if hasattr(obj, "__slots__"):
        for s in obj.__slots__:
            try:
                v = getattr(obj, s)
            except Exception:
                continue
            size += sizeof(v, _seen=_seen)

    return size

def profile_object_members(obj, *, sort_by="size", topk=None):
    """
    Print memory usage per non-None member of an object.
    """
    rows = []

    # __dict__ members
    if hasattr(obj, "__dict__"):
        for name, val in obj.__dict__.items():
            if val is None:
                continue
            size = sizeof(val)
            rows.append((name, type(val).__name__, size))

    # __slots__ members
    if hasattr(obj, "__slots__"):
        for name in obj.__slots__:
            try:
                val = getattr(obj, name)
            except Exception:
                continue
            if val is None:
                continue
            size = sizeof(val)
            rows.append((name, type(val).__name__, size))

    if sort_by == "size":
        rows.sort(key=lambda x: x[2], reverse=True)

    total = sum(r[2] for r in rows)

    print(f"{'member':30s} {'type':20s} {'size (KB)':>12s}")
    print("-" * 70)

    for name, typ, size in rows[:topk]:
        print(f"{name:30s} {typ:20s} {size/1024:12.2f}")

    print("-" * 70)
    print(f"{'TOTAL':30s} {'':20s} {total/1024:12.2f} KB")

    return rows