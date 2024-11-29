# Copyright (c) Opendatalab. All rights reserved.
import torch
import gc


def clean_memory():
    """
    清理内存中的未使用数据。

    该函数根据当前环境执行不同的内存清理操作：
    - 如果PyTorch库可用且环境支持CUDA，则优先清理GPU内存。
    - 无论环境是否支持CUDA，都会执行Python的垃圾回收机制以清理内存。

    此函数没有输入参数和返回值。
    """
    # 检查CUDA是否可用，如果可用，则执行GPU内存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 清空CUDA缓存中的未使用数据
        torch.cuda.ipc_collect()  # 收集并清理CUDA的IPC（进程间通信）资源

    # 执行Python的垃圾回收，清理所有不再使用的对象
    gc.collect()