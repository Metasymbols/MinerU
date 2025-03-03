from magic_pdf.operators.models import InferenceResult
from magic_pdf.model.model_list import MODEL
from magic_pdf.libs.config_reader import (get_device, get_formula_config,
                                          get_layout_config,
                                          get_local_models_dir,
                                          get_table_recog_config)
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.data.dataset import Dataset
import magic_pdf.model as model_config
from magic_pdf.model.sub_modules.model_utils import get_vram
from magic_pdf.model.batch_analyze import BatchAnalyze
from loguru import logger
import paddle
import os
import time

import torch

os.environ['FLAGS_npu_jit_compile'] = '0'  # 关闭paddle的jit编译
os.environ['FLAGS_use_stride_kernel'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 让mps可以fallback
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新
# 关闭paddle的信号处理

paddle.disable_signal_handler()


try:
    import torchtext
    if torchtext.__version__ >= '0.18.0':
        torchtext.disable_torchtext_deprecation_warning()
except ImportError:
    pass


class ModelSingleton:
    _instance = None
    _models = {}
    _lock = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            import threading
            with threading.Lock():
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._lock = threading.Lock()
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def get_model(
        self,
        ocr: bool,
        show_log: bool,
        lang=None,
        layout_model=None,
        formula_enable=None,
        table_enable=None,
    ):
        key = (ocr, show_log, lang, layout_model, formula_enable, table_enable)
        with self._lock:
            if key not in self._models:
                try:
                    self._models[key] = custom_model_init(
                        ocr=ocr,
                        show_log=show_log,
                        lang=lang,
                        layout_model=layout_model,
                        formula_enable=formula_enable,
                        table_enable=table_enable,
                    )
                except Exception as e:
                    logger.error(f"Model initialization failed: {str(e)}")
                    raise
            return self._models[key]

    def clear_models(self):
        """清理所有已加载的模型"""
        with self._lock:
            for model in self._models.values():
                try:
                    if hasattr(model, 'cleanup'):
                        model.cleanup()
                except Exception as e:
                    logger.warning(f"Model cleanup failed: {str(e)}")
            self._models.clear()

    def remove_model(self, key):
        """移除指定的模型"""
        with self._lock:
            if key in self._models:
                try:
                    model = self._models[key]
                    if hasattr(model, 'cleanup'):
                        model.cleanup()
                    del self._models[key]
                except Exception as e:
                    logger.warning(f"Model removal failed: {str(e)}")


def custom_model_init(
    ocr: bool = False,
    show_log: bool = False,
    lang=None,
    layout_model=None,
    formula_enable=None,
    table_enable=None,
):

    model = None

    if model_config.__model_mode__ == 'lite':
        logger.warning(
            'The Lite mode is provided for developers to conduct testing only, and the output quality is '
            'not guaranteed to be reliable.'
        )
        model = MODEL.Paddle
    elif model_config.__model_mode__ == 'full':
        model = MODEL.PEK

    if model_config.__use_inside_model__:
        model_init_start = time.time()
        if model == MODEL.Paddle:
            from magic_pdf.model.pp_structure_v2 import CustomPaddleModel

            custom_model = CustomPaddleModel(
                ocr=ocr, show_log=show_log, lang=lang)
        elif model == MODEL.PEK:
            from magic_pdf.model.pdf_extract_kit import CustomPEKModel

            # 从配置文件读取model-dir和device
            local_models_dir = get_local_models_dir()
            device = get_device()

            layout_config = get_layout_config()
            if layout_model is not None:
                layout_config['model'] = layout_model

            formula_config = get_formula_config()
            if formula_enable is not None:
                formula_config['enable'] = formula_enable

            table_config = get_table_recog_config()
            if table_enable is not None:
                table_config['enable'] = table_enable

            model_input = {
                'ocr': ocr,
                'show_log': show_log,
                'models_dir': local_models_dir,
                'device': device,
                'table_config': table_config,
                'layout_config': layout_config,
                'formula_config': formula_config,
                'lang': lang,
            }

            custom_model = CustomPEKModel(**model_input)
        else:
            logger.error('Not allow model_name!')
            exit(1)
        model_init_cost = time.time() - model_init_start
        logger.info(f'model init cost: {model_init_cost}')
    else:
        logger.error(
            'use_inside_model is False, not allow to use inside model')
        exit(1)

    return custom_model


def doc_analyze(
    dataset: Dataset,
    ocr: bool = False,
    show_log: bool = False,
    start_page_id=0,
    end_page_id=None,
    lang=None,
    layout_model=None,
    formula_enable=None,
    table_enable=None,
) -> InferenceResult:
    try:
        end_page_id = (
            end_page_id
            if end_page_id is not None and end_page_id >= 0
            else len(dataset) - 1
        )

        model_manager = ModelSingleton()
        custom_model = model_manager.get_model(
            ocr, show_log, lang, layout_model, formula_enable, table_enable
        )

        batch_analyze = False
        batch_ratio = 1
        device = get_device()

        # 检查设备支持
        npu_support = False
        if str(device).startswith("npu"):
            import torch_npu
            if torch_npu.npu.is_available():
                npu_support = True

        # 动态调整batch size
        if torch.cuda.is_available() and device != 'cpu' or npu_support:
            try:
                gpu_memory = int(
                    os.getenv("VIRTUAL_VRAM_SIZE", round(get_vram(device))))
                if gpu_memory is not None and gpu_memory >= 8:
                    # 根据显存大小动态调整batch_ratio
                    batch_ratio = max(1, min(32, 2 ** (int(gpu_memory / 8))))
                    logger.info(
                        f'GPU内存: {gpu_memory} GB, 批处理比例: {batch_ratio}')
                    batch_analyze = True
            except Exception as e:
                logger.warning(f"获取GPU内存失败: {str(e)}，使用单页面处理模式")
                batch_analyze = False

        model_json = []
        doc_analyze_start = time.time()
        total_pages = end_page_id - start_page_id + 1

        try:
            if batch_analyze:
                # 批量处理
                images = []
                page_wh_list = []
                for index in range(len(dataset)):
                    if start_page_id <= index <= end_page_id:
                        page_data = dataset.get_page(index)
                        img_dict = page_data.get_image()
                        images.append(img_dict['img'])
                        page_wh_list.append(
                            (img_dict['width'], img_dict['height']))

                # 创建批处理分析器并执行
                batch_model = BatchAnalyze(
                    model=custom_model, batch_ratio=batch_ratio)
                analyze_result = batch_model(images)

                # 处理结果
                for index in range(len(dataset)):
                    if start_page_id <= index <= end_page_id:
                        result = analyze_result.pop(0)
                        page_width, page_height = page_wh_list.pop(0)
                    else:
                        result = []
                        page_height = 0
                        page_width = 0

                    page_info = {'page_no': index,
                                 'width': page_width, 'height': page_height}
                    page_dict = {'layout_dets': result, 'page_info': page_info}
                    model_json.append(page_dict)

            else:
                # 单页面处理
                for index in range(len(dataset)):
                    page_data = dataset.get_page(index)
                    img_dict = page_data.get_image()
                    img = img_dict['img']
                    page_width = img_dict['width']
                    page_height = img_dict['height']

                    if start_page_id <= index <= end_page_id:
                        page_start = time.time()
                        try:
                            result = custom_model(img)
                            page_time = round(time.time() - page_start, 2)
                            logger.info(f'页面 {index} 处理完成，耗时: {page_time}秒')
                        except Exception as e:
                            logger.error(f"处理页面 {index} 时发生错误: {str(e)}")
                            result = []
                    else:
                        result = []

                    page_info = {'page_no': index,
                                 'width': page_width, 'height': page_height}
                    page_dict = {'layout_dets': result, 'page_info': page_info}
                    model_json.append(page_dict)

        except Exception as e:
            logger.error(f"文档分析过程中发生错误: {str(e)}")
            raise

        finally:
            # 清理内存
            gc_start = time.time()
            clean_memory(device)
            gc_time = round(time.time() - gc_start, 2)
            logger.info(f'内存清理耗时: {gc_time}秒')

        # 计算并记录性能指标
        doc_analyze_time = round(time.time() - doc_analyze_start, 2)
        doc_analyze_speed = round(total_pages / doc_analyze_time, 2)
        logger.info(
            f'文档分析总耗时: {doc_analyze_time}秒, '
            f'处理速度: {doc_analyze_speed} 页/秒, '
            f'总页数: {total_pages}'
        )

        return InferenceResult(model_json, dataset)

    except Exception as e:
        logger.error(f"文档分析失败: {str(e)}")
        raise
