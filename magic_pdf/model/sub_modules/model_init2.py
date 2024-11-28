#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model_init2.py
@Time    :   2024/11/28 13:10:19
@Author  :   若水紫风 
@Contact :   XXXXXXXXX@qq.com
@License :   木兰公共协议 MulanPubL-2.0 版
@Version :   1.0
'''
'''

'''

from functools import lru_cache
from typing import Dict, Optional

# here put the import lib
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
    def __init__(self, model_path, max_time, max_new_tokens=2048):
        self.model = StructTableModel(model_path, max_new_tokens, max_time)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


class TableMasterPaddleModelWrapper(BaseModel):
    def __init__(self, config):
        self.model = TableMasterPaddleModel(config)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


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
        return self.model.predict(*args, **kwargs)


class DocLayoutYOLOModelWrapper(BaseModel):
    def __init__(self, weight:str, device='cpu'):
        if not weight:
            raise ValueError("weight cannot be empty")
        self.model = DocLayoutYOLOModel(weight, device)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


class ModifiedPaddleOCRWrapper(BaseModel):
    def __init__(self, show_log=False, det_db_box_thresh=0.3, lang=None, use_dilation=True, det_db_unclip_ratio=1.8):
        self.model = ModifiedPaddleOCR(
            show_log, det_db_box_thresh, lang, use_dilation, det_db_unclip_ratio)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


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
            AtomicModel.Layout: ModelFactory._create_doclayout_yolo_model,
        }

        create_func = model_mapping.get(model_name)
        if create_func:
            return create_func(**kwargs)
        else:
            logger.error(f"Unknown model type: {model_name}")
            raise ValueError(f"Unknown model type: {model_name}")

    @staticmethod
    def _create_layout_model(**kwargs):
        if model_name == AtomicModel.Layout:
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
            kwargs.get("ocr_show_log", False),
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

        table_model_mapping = {
            MODEL_NAME.STRUCT_EQTABLE: StructTableModelWrapper,
            MODEL_NAME.TABLE_MASTER: TableMasterPaddleModelWrapper,
            MODEL_NAME.RAPID_TABLE: RapidTableModelWrapper,
        }

        model_cls = table_model_mapping.get(table_model_type)
        if model_cls:
            return model_cls(model_path, max_time) if model_cls == StructTableModelWrapper else model_cls({'model_dir': model_path, 'device': device})
        else:
            logger.error("Invalid table model type")
            raise ValueError("Invalid table model type")

    @staticmethod
    def _create_doclayout_yolo_model(**kwargs):
        return DocLayoutYOLOModelWrapper(
            kwargs.get("doclayout_yolo_weights"), 
            kwargs.get("device"))


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
    # pass
 
   
    singleton = AtomModelSingleton()

    # 模拟获取不同的模型
    # layout_model = singleton.get_atom_model(
    #     atom_model_name= AtomicModel.Layout,
    #     layout_model_name=MODEL_NAME.LAYOUTLMv3,
    #     layout_weights=r"D:\ProgramData\.cache\hub\opendatalab\PDF-Extract-Kit-1___0\models\Layout\LayoutLMv3\model_final.pth",
    #     doclayout_yolo_weights=r"D:\ProgramData\.cache\hub\opendatalab\PDF-Extract-Kit-1___0\models\Layout\YOLO\doclayout_yolo_ft.pt",
    #     layout_config_file=r"G:\Workspace\code\py_code\MinerU\magic_pdf\resources\model_config\layoutlmv3\layoutlmv3_base_inference.yaml",
    #     device="cpu")
    # # 加载图像
    # import cv2
    # image = cv2.imread(r"G:\Workspace\code\py_code\MinerU\demo\demo2_页面_1.png")
    # print(layout_model.predict(image))
    
    #######################
    mfd_model = singleton.get_atom_model(
        atom_model_name=AtomicModel.MFD, 
        mfd_weights=r"D:\ProgramData\.cache\hub\opendatalab\PDF-Extract-Kit-1___0\models\MFD\YOLO\yolo_v8_ft.pt", 
        device="cpu")
    print(mfd_model)