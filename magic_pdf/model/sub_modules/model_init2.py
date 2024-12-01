
from functools import lru_cache
from typing import Dict, Optional

from loguru import logger
from magic_pdf.libs.Constants import MODEL_NAME
from magic_pdf.model.model_list import AtomicModel
from magic_pdf.model.sub_modules.layout.doclayout_yolo.DocLayoutYOLO import \
    DocLayoutYOLOModel
from magic_pdf.model.sub_modules.layout.layoutlmv3.model_init import \
    Layoutlmv3_Predictor
from magic_pdf.model.sub_modules.mfd.yolov8.YOLOv8 import YOLOv8MFDModel
from magic_pdf.model.sub_modules.mfr.unimernet.Unimernet import UnimernetModel
from magic_pdf.model.sub_modules.ocr.paddleocr.ppocr_273_mod import \
    ModifiedPaddleOCR
from magic_pdf.model.sub_modules.table.rapidtable.rapid_table import \
    RapidTableModel
from magic_pdf.model.sub_modules.table.structeqtable.struct_eqtable import \
    StructTableModel
from magic_pdf.model.sub_modules.table.tablemaster.tablemaster_paddle import \
    TableMasterPaddleModel


class BaseModel:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")


# 各个具体模型的封装类
class StructTableModelWrapper(BaseModel):
    def __init__(self, model_path, max_time, max_new_tokens=1024):
        self.model = StructTableModel(model_path, max_new_tokens, max_time)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


class TableMasterPaddleModelWrapper(BaseModel):
    def __init__(self, config):
        self.model = TableMasterPaddleModel(config)

    def predict(self, *args, **kwargs):
        return self.model.img2html(*args, **kwargs)


class RapidTableModelWrapper(BaseModel):
    def __init__(self):
        self.model = RapidTableModel()

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


class YOLOv8MFDModelWrapper(BaseModel):
    def __init__(self, weight, device='cpu'):
        self.model = YOLOv8MFDModel(weight, device)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


class UnimernetModelWrapper(BaseModel):
    def __init__(self, weight_dir, cfg_path, device='cpu'):
        self.model = UnimernetModel(weight_dir, cfg_path, device)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


class Layoutlmv3_PredictorWrapper(BaseModel):
    def __init__(self, weight, config_file, device):
        self.model = Layoutlmv3_Predictor(weight, config_file, device)

    def predict(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class DocLayoutYOLOModelWrapper(BaseModel):
    def __init__(self, weight: str, device='cpu'):
        if not weight:
            raise ValueError("weight cannot be empty")
        self.model = DocLayoutYOLOModel(weight, device)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


class ModifiedPaddleOCRWrapper(BaseModel):
    def __init__(self,
                 show_log=False,
                 det_db_box_thresh=0.3,
                 lang=None,
                 use_dilation=True,
                 det_db_unclip_ratio=1.8):
        if lang is not None and lang != '':
            self.model = ModifiedPaddleOCR(
                show_log=show_log,
                det_db_box_thresh=det_db_box_thresh,
                lang=lang,
                use_dilation=use_dilation,
                det_db_unclip_ratio=det_db_unclip_ratio,)
        else:
            self.model = ModifiedPaddleOCR(
                show_log=show_log,
                det_db_box_thresh=det_db_box_thresh,
                # lang=lang,
                # use_dilation=use_dilation,
                det_db_unclip_ratio=det_db_unclip_ratio,)

    def predict(self, *args, **kwargs):
        return self.model.ocr(*args, **kwargs)


# 简化的工厂模式
class ModelFactory:
    @staticmethod
    def create_model(model_name: str, **kwargs) -> BaseModel:
        """ 根据模型类型创建不同的模型实例 """
        model_mapping = {
            AtomicModel.Layout: ModelFactory._create_layout_model,
            AtomicModel.MFD: ModelFactory._create_mfd_model,
            AtomicModel.MFR: ModelFactory._create_mfr_model,
            AtomicModel.OCR: ModelFactory._create_ocr_model,
            AtomicModel.Table: ModelFactory._create_table_model,
            # AtomicModel.Layout: ModelFactory._create_doclayout_yolo_model,
        }

        create_func = model_mapping.get(model_name)
        if create_func:
            return create_func(**kwargs)
        else:
            logger.error(f"Unknown model type: {model_name}")
            raise ValueError(f"Unknown model type: {model_name}")

    @staticmethod
    def _create_layout_model(**kwargs):

        layout_model_name = kwargs.get("layout_model_name")
        if layout_model_name == MODEL_NAME.LAYOUTLMv3:
            return Layoutlmv3_PredictorWrapper(
                kwargs.get("layout_weights", ""),
                kwargs.get("layout_config_file", ""),
                kwargs.get("device")
            )
        elif layout_model_name == MODEL_NAME.DocLayout_YOLO:
            return DocLayoutYOLOModelWrapper(
                kwargs.get("doclayout_yolo_weights"),
                kwargs.get("device")
            )

    @staticmethod
    def _create_mfd_model(**kwargs):
        return YOLOv8MFDModelWrapper(
            kwargs.get("mfd_weights"),
            kwargs.get("device"))

    @staticmethod
    def _create_mfr_model(**kwargs):
        return UnimernetModelWrapper(
            kwargs.get("mfr_weight_dir"),
            kwargs.get("mfr_cfg_path"),
            kwargs.get("device"))

    @staticmethod
    def _create_ocr_model(**kwargs):
        return ModifiedPaddleOCRWrapper(
            kwargs.get("show_log", False),
            kwargs.get("det_db_box_thresh", 0.3),
            kwargs.get("lang"),
            kwargs.get("use_dilation", True),
            kwargs.get("det_db_unclip_ratio", 1.8)
        )

    @staticmethod
    def _create_table_model(**kwargs):
        table_model_type = kwargs.get("table_model_name")
        model_path = kwargs.get("table_model_path")
        max_time = kwargs.get("table_max_time", 60)
        device = kwargs.get("device", "cpu")

        if table_model_type == MODEL_NAME.STRUCT_EQTABLE:
            table_model = StructTableModelWrapper(
                model_path, max_new_tokens=2048, max_time=max_time)
        elif table_model_type == MODEL_NAME.TABLE_MASTER:
            table_model = TableMasterPaddleModelWrapper({
                'model_dir': model_path,
                'device': device
            })
        elif table_model_type == MODEL_NAME.RAPID_TABLE:
            table_model = RapidTableModelWrapper()
        else:
            logger.error('table model type not allowed')

        return table_model


# 缓存优化：使用 lru_cache 来缓存模型实例，减少重复计算
@lru_cache(maxsize=128)
def get_cached_model(model_name: str, **kwargs):
    return ModelFactory.create_model(model_name, **kwargs)


# 单例模式：确保只存在一个 AtomModelSingleton 实例
class AtomModelSingleton:
    _instance: Optional['AtomModelSingleton'] = None
    _models: Dict[str, BaseModel] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_atom_model(self, atom_model_name: str, **kwargs) -> BaseModel:
        lang = kwargs.get("lang", None)
        layout_model_name = kwargs.get("layout_model_name", None)

        model_key = (atom_model_name, layout_model_name, lang)

        if model_key not in self._models:
            # 使用缓存的方式来优化模型初始化
            self._models[model_key] = get_cached_model(
                model_name=atom_model_name, **kwargs)
        return self._models[model_key]


# 测试示例
if __name__ == "__main__":
    # # 加载图像
    import cv2
    image = cv2.imread(r"G:\Workspace\code\py_code\MinerU\demo\demo2_页面_5.png")

    singleton = AtomModelSingleton()

    # 模拟获取不同的模型
    layout_model = singleton.get_atom_model(
        atom_model_name=AtomicModel.Layout,
        layout_model_name=MODEL_NAME.LAYOUTLMv3,
        layout_weights=r"D:\ProgramData\.cache\hub\opendatalab\PDF-Extract-Kit-1___0\models\Layout\LayoutLMv3\model_final.pth",
        layout_config_file=r"G:\Workspace\code\py_code\MinerU\magic_pdf\resources\model_config\layoutlmv3\layoutlmv3_base_inference.yaml",
        device="cpu")
    print(layout_model.predict(image))

    # ##############
    # layout_model = singleton.get_atom_model(
    #     atom_model_name= AtomicModel.Layout,
    #     layout_model_name=MODEL_NAME.DocLayout_YOLO,
    #     doclayout_yolo_weights=r"D:\ProgramData\.cache\hub\opendatalab\PDF-Extract-Kit-1___0\models\Layout\YOLO\doclayout_yolo_ft.pt",
    #     device="cpu")
    # print(layout_model.predict(image))

    # #######################
    # mfd_model = singleton.get_atom_model(
    #     atom_model_name=AtomicModel.MFD,
    #     mfd_weights=r"D:\ProgramData\.cache\hub\opendatalab\PDF-Extract-Kit-1___0\models\MFD\YOLO\yolo_v8_ft.pt",
    #     device="cpu")
    # print(mfd_model.predict(image))

    # # ##############
    # mfr_res = mfd_model.predict(image)
    # mfr_model = singleton.get_atom_model(
    #     atom_model_name=AtomicModel.MFR,
    #     mfr_weight_dir=r"D:\ProgramData\.cache\hub\opendatalab\PDF-Extract-Kit-1___0\models\MFR\unimernet_small",
    #     mfr_cfg_path=r"G:\Workspace\code\py_code\MinerU\magic_pdf\resources\model_config\UniMERNet\demo.yaml",
    #     device="cpu")
    # print(mfr_model.predict(mfr_res,image))

    # #################
    # # cn 中文 en 英文
    # ocr_model = singleton.get_atom_model(
    #     atom_model_name=AtomicModel.OCR,
    #     lang="en"
    #     )
    # print(ocr_model.predict(image))

    # #################
    # # struct_eqtable 必须有CUDA才可以调用
    # table_model = singleton.get_atom_model(
    #     atom_model_name=AtomicModel.Table,
    #     table_model_name=MODEL_NAME.STRUCT_EQTABLE,
    #     table_model_path=r"D:\ProgramData\.cache\hub\opendatalab\PDF-Extract-Kit-1___0\models\TabRec\StructEqTable"
    #     )
    # print(table_model.predict(image))

    # ###############
    # table_model = singleton.get_atom_model(
    #     atom_model_name=AtomicModel.Table,
    #     table_model_name=MODEL_NAME.TABLE_MASTER,
    #     table_model_path=r"D:\ProgramData\.cache\hub\opendatalab\PDF-Extract-Kit-1___0\models\TabRec\TableMaster"
    #     )
    # print(table_model.predict(image))

    # ###########################
    # table_model = singleton.get_atom_model(
    #     atom_model_name=AtomicModel.Table,
    #     table_model_name=MODEL_NAME.RAPID_TABLE,
    #     )
    # print(table_model.predict(image))
