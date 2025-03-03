from typing import Dict, Optional
from magic_pdf.model.magic_model import MagicModel
from magic_pdf.model.model_list import get_model_list
from magic_pdf.model.sub_modules.model_init import init_model


class ModelManager:
    """模型管理器，负责管理和初始化PDF解析所需的各种模型。
    
    此类主要负责：
    1. 模型的初始化和加载
    2. 模型配置的管理
    3. 模型资源的释放
    
    Attributes:
        models (Dict[str, MagicModel]): 已加载的模型字典
        device (str): 设备类型（'cpu'或'cuda'）
    """
    
    def __init__(self, device: str = 'cpu'):
        """初始化模型管理器。

        Args:
            device (str, optional): 设备类型. Defaults to 'cpu'.
        """
        self.models = {}
        self.device = device
    
    def load_model(self, model_name: str) -> Optional[MagicModel]:
        """加载指定的模型。

        Args:
            model_name (str): 模型名称

        Returns:
            Optional[MagicModel]: 加载的模型对象，如果加载失败则返回None
        """
        if model_name in self.models:
            return self.models[model_name]
        
        model_list = get_model_list()
        if model_name not in model_list:
            return None
        
        model = init_model(model_name, self.device)
        if model is not None:
            self.models[model_name] = model
        
        return model
    
    def get_model(self, model_name: str) -> Optional[MagicModel]:
        """获取已加载的模型。

        Args:
            model_name (str): 模型名称

        Returns:
            Optional[MagicModel]: 模型对象，如果模型未加载则返回None
        """
        return self.models.get(model_name)
    
    def release_model(self, model_name: str) -> bool:
        """释放指定的模型。

        Args:
            model_name (str): 模型名称

        Returns:
            bool: 释放是否成功
        """
        if model_name in self.models:
            model = self.models.pop(model_name)
            model.release()
            return True
        return False
    
    def release_all(self):
        """释放所有已加载的模型。"""
        for model_name in list(self.models.keys()):
            self.release_model(model_name)