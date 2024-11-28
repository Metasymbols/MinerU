#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   模型文件映射到内存.py
@Time    :   2024/11/27 10:22:00
@Author  :   若水紫风 
@Contact :   XXXXXXXXX@qq.com
@License :   木兰公共协议 MulanPubL-2.0 版
@Version :   1.0
'''
'''
用于加载管理模型文件映射到内存的模块

'''

# here put the import lib




from pathlib import Path
import concurrent.futures
import json
import mmap
import os
import threading
from abc import ABC, abstractmethod
from contextlib import ExitStack
from typing import Any, Dict, List, Optional
from loguru import logger
class ModelLoader(ABC):
    """
    ModelLoader 是一个抽象基类，定义了加载模型的基本接口。
    """

    @abstractmethod
    def load_model(self) -> Optional[List[mmap.mmap]]:
        """
        将模型文件加载到内存中。

        :return: 内存映射对象列表。
        """
        pass

    @abstractmethod
    def save_meta_info(self, meta_info: Dict[str, Any]) -> None:
        """
        保存模型文件的元信息。

        :param meta_info: 包含元信息的字典。
        """
        pass

    @abstractmethod
    def load_meta_info(self) -> Optional[Dict[str, Any]]:
        """
        加载模型文件的元信息。

        :return: 包含元信息的字典。
        """
        pass


class MultiFileModelLoader(ModelLoader):
    """
    MultiFileModelLoader 是 ModelLoader 的具体实现，
    使用内存映射文件技术加载多个模型文件。
    """

    def __init__(
        self, model_file_paths: List[str],
        config_path: Path = Path(
            "G:\\Workspace\\code\\py_code\\MinerU\\magic_pdf\\workbench\\file_map_to_menory\\config.json")
    ):
        """
        初始化模型加载器。

        :param model_file_paths: 模型文件路径列表。
        :param config_path: 配置文件路径，默认为 'config.json'。
        """
        self.model_file_paths: List[str] = model_file_paths
        self.config_path: Path = config_path
        self.config = self._load_config()
        self.meta_info_file_path: Path = self.config['meta_info_file_path']
        self.meta_info_manager: MetaInfoManager = MetaInfoManager(
            self.meta_info_file_path)

    def _load_config(self) -> Dict[str, Any]:
        """
        从 JSON 文件中加载配置。

        :return: 包含配置的字典。
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                logger.info("配置加载成功")
                return config
        except (IOError, OSError, ValueError) as e:
            logger.error(f"加载配置失败: {e}")
            raise

    def load_model(self) -> Optional[List[mmap.mmap]]:
        """
        将模型文件加载到内存中。

        :return: 内存映射对象列表。
        """
        try:
            meta_info: Optional[Dict[str, Any]] = self.meta_info_manager.load()
            if not meta_info or self._is_meta_info_outdated(meta_info):
                # 如果元信息不存在或已过期，重新生成并保存
                meta_info = self._generate_meta_info()
                self.meta_info_manager.save(meta_info)

            memory_maps: List[mmap.mmap] = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(
                    self._load_model_file, file_path, meta_info) for file_path in self.model_file_paths]
                for future in concurrent.futures.as_completed(futures):
                    mm = future.result()
                    if mm:
                        memory_maps.append(mm)
            logger.info("模型加载成功")
            return memory_maps
        except (IOError, OSError, ValueError) as e:
            logger.error(f"加载模型失败: {e}")
            return None

    def _load_model_file(self, file_path: Path, meta_info: Dict[str, Any]) -> Optional[mmap.mmap]:
        """
        加载单个模型文件。

        :param file_path: 模型文件路径。
        :param meta_info: 包含元信息的字典。
        :return: 内存映射对象。
        """
        try:
            with open(file_path, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), meta_info[file_path]['file_size'])
                logger.info(f"{file_path} 加载成功")
                return mm
        except (IOError, OSError, ValueError) as e:
            logger.error(f"加载 {file_path} 失败: {e}")
            return None

    def _generate_meta_info(self) -> Dict[str, Any]:
        """
        生成模型文件的元信息。

        :return: 包含元信息的字典。
        """
        meta_info: Dict[str, Any] = {}
        for file_path in self.model_file_paths:
            file_name = os.path.basename(file_path)
            meta_info[file_path] = {
                'model_name': file_name,
                'file_size': os.path.getsize(file_path),
                'last_modified': os.path.getmtime(file_path)
            }
        logger.info("元信息生成成功")
        return meta_info

    def _is_meta_info_outdated(self, meta_info: Dict[str, Any]) -> bool:
        """
        检查元信息是否已过期。

        :param meta_info: 包含元信息的字典。
        :return: 如果元信息已过期返回 True，否则返回 False。
        """
        for file_path in self.model_file_paths:
            current_last_modified: float = os.path.getmtime(file_path)
            if meta_info[file_path]['last_modified'] != current_last_modified:
                logger.info(f"{file_path} 的元信息已过期")
                return True
        return False

    def save_meta_info(self, meta_info: Dict[str, Any]) -> None:
        """
        保存模型文件的元信息。

        :param meta_info: 包含元信息的字典。
        """
        try:
            self.meta_info_manager.save(meta_info)
            logger.info("元信息保存成功")
        except (IOError, OSError, ValueError) as e:
            logger.error(f"保存元信息失败: {e}")

    def load_meta_info(self) -> Optional[Dict[str, Any]]:
        """
        加载模型文件的元信息。

        :return: 包含元信息的字典。
        """
        try:
            meta_info = self.meta_info_manager.load()
            if meta_info:
                logger.info("元信息加载成功")
            else:
                logger.warning("元信息文件未找到")
            return meta_info
        except (IOError, OSError, ValueError) as e:
            logger.error(f"加载元信息失败: {e}")
            return None


class MetaInfoManager:
    """
    MetaInfoManager 管理模型文件的元信息，包括序列化和反序列化。
    """

    def __init__(self, meta_info_file_path: str):
        """
        初始化元信息管理器。

        :param meta_info_file_path: 元信息文件路径。
        """
        self.meta_info_file_path: str = meta_info_file_path

    def save(self, meta_info: Dict[str, Any]) -> None:
        """
        将元信息保存到文件。

        :param meta_info: 包含元信息的字典。
        """
        try:
            with open(self.meta_info_file_path, 'w') as f:
                json.dump(meta_info, f)
        except (IOError, OSError, ValueError) as e:
            logger.error(f"保存元信息失败: {e}")

    def load(self) -> Optional[Dict[str, Any]]:
        """
        从文件加载元信息。

        :return: 包含元信息的字典，如果文件不存在则返回 None。
        """
        try:
            if not os.path.exists(self.meta_info_file_path):
                logger.warning("元信息文件未找到")
                return None
            with open(self.meta_info_file_path, 'r') as f:
                return json.load(f)
        except (IOError, OSError, ValueError) as e:
            logger.error(f"加载元信息失败: {e}")
            return None


if __name__ == '__main__':
    model_file_paths: List[str] = [
        'D:\ProgramData\.cache\hub\opendatalab\PDF-Extract-Kit-1___0\models\MFD\YOLO\yolo_v8_ft.pt'
    ]
    loader: MultiFileModelLoader = MultiFileModelLoader(model_file_paths)

    # 加载模型
    models: Optional[List[mmap.mmap]] = loader.load_model()
    if models:
        print("模型加载成功")
        # 关闭内存映射文件
        print(models)
        for model in models:
            model.close()
    else:
        print("加载模型失败")

    # 获取元信息
    meta_info: Optional[Dict[str, Any]] = loader.load_meta_info()
    print("元信息:", meta_info)
