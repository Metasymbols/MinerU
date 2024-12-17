from ultralytics import YOLO
from PIL import Image
from typing import Union, Any
import numpy as np


class YOLOv8MFDModel:
    """YOLOv8 模型封装类，用于多字段检测(MFD)任务"""
    
    def __init__(self, weight: str, device: str = 'cpu') -> None:
        """
        初始化YOLOv8模型
        
        Args:
            weight: 模型权重文件路径
            device: 运行设备，默认为'cpu'，可选'cuda'等
        """
        self.mfd_model = YOLO(weight)
        self.device = device

    def predict(self, image: Union[str, Image.Image, np.ndarray], 
                conf: float = 0.25,
                iou: float = 0.45,
                imgsz: int = 1888) -> Any:
        """
        执行目标检测预测
        
        Args:
            image: 输入图像，支持PIL Image、numpy数组或图像路径
            conf: 置信度阈值
            iou: IoU阈值
            imgsz: 输入图像大小
            
        Returns:
            检测结果
        """
        mfd_res = self.mfd_model.predict(
            image,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,  # 减少不必要的输出
            device=self.device
        )[0]
        return mfd_res
