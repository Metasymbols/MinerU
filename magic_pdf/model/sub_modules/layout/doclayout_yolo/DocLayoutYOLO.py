from doclayout_yolo import YOLOv10


class DocLayoutYOLOModel(object):
    """文档布局检测模型类

    使用YOLOv10模型进行文档布局检测，可以识别文档中的不同区域类型。

    Args:
        weight (str): 模型权重文件路径
        device (str): 运行设备，如'cpu'或'cuda'
    """
    def __init__(self, weight, device):
        self.model = YOLOv10(weight)
        self.device = device

    def predict(self, image):
        """对单张图像进行布局检测

        Args:
            image: 输入图像，支持numpy数组格式

        Returns:
            list: 检测结果列表，每个元素为字典，包含以下字段：
                - category_id (int): 区域类型ID
                - poly (list): 区域多边形坐标，格式为[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]
                - score (float): 检测置信度分数
        """
        layout_res = []
        doclayout_yolo_res = self.model.predict(
            image,
            imgsz=1280,
            conf=0.10,
            iou=0.45,
            verbose=False, device=self.device
        )[0]
        for xyxy, conf, cla in zip(
            doclayout_yolo_res.boxes.xyxy.cpu(),
            doclayout_yolo_res.boxes.conf.cpu(),
            doclayout_yolo_res.boxes.cls.cpu(),
        ):
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
            new_item = {
                "category_id": int(cla.item()),
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": round(float(conf.item()), 3),
            }
            layout_res.append(new_item)
        return layout_res

    def batch_predict(self, images: list, batch_size: int) -> list:
        """批量对图像进行布局检测

        Args:
            images (list): 输入图像列表，每个元素为numpy数组格式的图像
            batch_size (int): 批处理大小

        Returns:
            list: 检测结果列表，每个元素为一张图像的检测结果列表，
                 检测结果格式同predict方法的返回值
        """
        images_layout_res = []
        for index in range(0, len(images), batch_size):
            doclayout_yolo_res = [
                image_res.cpu()
                for image_res in self.model.predict(
                    images[index : index + batch_size],
                    imgsz=1280,
                    conf=0.10,
                    iou=0.45,
                    verbose=False,
                    device=self.device,
                )
            ]
            for image_res in doclayout_yolo_res:
                layout_res = []
                for xyxy, conf, cla in zip(
                    image_res.boxes.xyxy,
                    image_res.boxes.conf,
                    image_res.boxes.cls,
                ):
                    xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                    new_item = {
                        "category_id": int(cla.item()),
                        "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                        "score": round(float(conf.item()), 3),
                    }
                    layout_res.append(new_item)
                images_layout_res.append(layout_res)

        return images_layout_res
