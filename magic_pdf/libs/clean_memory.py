# Copyright (c) Opendatalab. All rights reserved.
import torch
import gc


def clean_memory(device='cuda'):
    """清理设备内存，释放未使用的缓存。

    根据设备类型执行相应的内存清理操作：
    - 对于CUDA设备，清空CUDA缓存并收集IPC内存
    - 对于NPU设备，清空NPU缓存
    - 对于MPS设备，清空MPS缓存
    最后执行Python垃圾回收

    Args:
        device (str, optional): 设备类型，可选值为'cuda'、'npu'或'mps'。默认为'cuda'。
    """
    if device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif str(device).startswith("npu"):
        import torch_npu
        if torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()
    elif str(device).startswith("mps"):
        torch.mps.empty_cache()
    gc.collect()