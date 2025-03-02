from ultralytics import YOLO


class YOLOv8MFDModel(object):
    """数学公式检测模型类

    使用YOLOv8模型进行数学公式区域检测，可以识别文档中的数学公式区域。

    Args:
        weight (str): 模型权重文件路径
        device (str): 运行设备，默认为'cpu'
    """
    def __init__(self, weight, device="cpu"):
        self.mfd_model = YOLO(weight)
        self.device = device

    def predict(self, image):
        """对单张图像进行数学公式区域检测

        Args:
            image: 输入图像，支持numpy数组格式

        Returns:
            object: YOLOv8检测结果对象，包含检测到的所有数学公式区域信息
        """
        mfd_res = self.mfd_model.predict(
            image, imgsz=1888, conf=0.25, iou=0.45, verbose=False, device=self.device
        )[0]
        return mfd_res

    def batch_predict(self, images: list, batch_size: int) -> list:
        """批量对图像进行数学公式区域检测

        Args:
            images (list): 输入图像列表，每个元素为numpy数组格式的图像
            batch_size (int): 批处理大小

        Returns:
            list: 检测结果列表，每个元素为一张图像的YOLOv8检测结果对象
        """
        images_mfd_res = []
        for index in range(0, len(images), batch_size):
            mfd_res = [
                image_res.cpu()
                for image_res in self.mfd_model.predict(
                    images[index : index + batch_size],
                    imgsz=1888,
                    conf=0.25,
                    iou=0.45,
                    verbose=False,
                    device=self.device,
                )
            ]
            for image_res in mfd_res:
                images_mfd_res.append(image_res)
        return images_mfd_res
