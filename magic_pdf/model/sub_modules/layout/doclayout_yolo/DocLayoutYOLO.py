from doclayout_yolo import YOLOv10


class DocLayoutYOLOModel:
    def __init__(self, weight: str, device: str):
        if not weight:
            raise ValueError("weight cannot be empty")
        self.model = YOLOv10(weight)
        self.device = device

    def predict(self, image):
        layout_res = []
        doclayout_yolo_res = self.model.predict(
            image,
            imgsz=1024,
            conf=0.25,
            iou=0.45,
            verbose=True,
            device=self.device
        )[0]
        boxes = doclayout_yolo_res.boxes
        for xyxy, conf, cla in zip(boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu()):
            xmin, ymin, xmax, ymax = map(lambda x: int(x.item()), xyxy)
            layout_res.append({
                'category_id': int(cla.item()),
                'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                'score': round(float(conf.item()), 3),
            })
        return layout_res
