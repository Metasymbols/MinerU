<<<<<<< HEAD
from magic_pdf.model.ppTableModel import ppTableModel
from magic_pdf.model.pek_sub_modules.structeqtable.StructTableModel import StructTableModel
from magic_pdf.model.pek_sub_modules.self_modify import ModifiedPaddleOCR
from magic_pdf.model.pek_sub_modules.post_process import get_croped_image, latex_rm_whitespace
from magic_pdf.model.pek_sub_modules.layoutlmv3.model_init import Layoutlmv3_Predictor
=======
import numpy as np
import torch
>>>>>>> eadf4ce7c3ac4d502ee626738b72da9b71819c4d
from loguru import logger
import os
import time
import cv2
import yaml
from PIL import Image

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新
os.environ['YOLO_VERBOSE'] = 'False'  # disable yolo logger

try:
    import torchtext

    if torchtext.__version__ >= "0.18.0":
        torchtext.disable_torchtext_deprecation_warning()
except ImportError:
    pass

<<<<<<< HEAD
except ImportError as e:
    logger.exception(e)
    logger.error(
        'Required dependency not installed, please install by \n'
        '"pip install magic-pdf[full] --extra-index-url https://myhloli.github.io/wheels/"')
    exit(1)


def table_model_init(table_model_type, model_path, max_time, _device_='cpu'):
    if table_model_type == STRUCT_EQTABLE:
        table_model = StructTableModel(
            model_path, max_time=max_time, device=_device_)
    else:
        config = {
            "model_dir": model_path,
            "device": _device_
        }
        table_model = ppTableModel(config)
    return table_model


def mfd_model_init(weight):
    mfd_model = YOLO(weight)
    return mfd_model


def mfr_model_init(weight_dir, cfg_path, _device_='cpu'):
    args = argparse.Namespace(cfg_path=cfg_path, options=None)
    cfg = Config(args)
    cfg.config.model.pretrained = os.path.join(weight_dir, "pytorch_model.bin")
    cfg.config.model.model_config.model_name = weight_dir
    cfg.config.model.tokenizer_config.path = weight_dir
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model = model.to(_device_)
    vis_processor = load_processor(
        'formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    mfr_transform = transforms.Compose([vis_processor, ])
    return [model, mfr_transform]


def layout_model_init(weight, config_file, device):
    model = Layoutlmv3_Predictor(weight, config_file, device)
    return model


def ocr_model_init(show_log: bool = False, det_db_box_thresh=0.3):
    model = ModifiedPaddleOCR(
        show_log=show_log, det_db_box_thresh=det_db_box_thresh)
    return model


class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # if not pil image, then convert to pil image
        if isinstance(self.image_paths[idx], str):
            raw_image = Image.open(self.image_paths[idx])
        else:
            raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
            return image


class AtomModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_atom_model(self, atom_model_name: str, **kwargs):
        if atom_model_name not in self._models:
            self._models[atom_model_name] = atom_model_init(
                model_name=atom_model_name, **kwargs)
        return self._models[atom_model_name]


def atom_model_init(model_name: str, **kwargs):

    if model_name == AtomicModel.Layout:
        atom_model = layout_model_init(
            kwargs.get("layout_weights"),
            kwargs.get("layout_config_file"),
            kwargs.get("device")
        )
    elif model_name == AtomicModel.MFD:
        atom_model = mfd_model_init(
            kwargs.get("mfd_weights")
        )
    elif model_name == AtomicModel.MFR:
        atom_model = mfr_model_init(
            kwargs.get("mfr_weight_dir"),
            kwargs.get("mfr_cfg_path"),
            kwargs.get("device")
        )
    elif model_name == AtomicModel.OCR:
        atom_model = ocr_model_init(
            kwargs.get("ocr_show_log"),
            kwargs.get("det_db_box_thresh")
        )
    elif model_name == AtomicModel.Table:
        atom_model = table_model_init(
            kwargs.get("table_model_type"),
            kwargs.get("table_model_path"),
            kwargs.get("table_max_time"),
            kwargs.get("device")
        )
    else:
        logger.error("model name not allow")
        exit(1)

    return atom_model
=======
from magic_pdf.libs.Constants import *
from magic_pdf.model.model_list import AtomicModel
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
from magic_pdf.model.sub_modules.model_utils import get_res_list_from_layout_res, crop_img, clean_vram
from magic_pdf.model.sub_modules.ocr.paddleocr.ocr_utils import get_adjusted_mfdetrec_res, get_ocr_result_list
>>>>>>> eadf4ce7c3ac4d502ee626738b72da9b71819c4d


class CustomPEKModel:

    def __init__(self, ocr: bool = False, show_log: bool = False, **kwargs):
        """
        ======== model init ========
        """
        # 获取当前文件（即 pdf_extract_kit.py）的绝对路径
        current_file_path = os.path.abspath(__file__)
        # 获取当前文件所在的目录(model)
        current_dir = os.path.dirname(current_file_path)
        # 上一级目录(magic_pdf)
        root_dir = os.path.dirname(current_dir)
        # model_config目录
        model_config_dir = os.path.join(root_dir, 'resources', 'model_config')
        # 构建 model_configs.yaml 文件的完整路径
        config_path = os.path.join(model_config_dir, 'model_configs.yaml')
        with open(config_path, "r", encoding='utf-8') as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
        # 初始化解析配置
<<<<<<< HEAD
        self.apply_layout = kwargs.get(
            "apply_layout", self.configs["config"]["layout"])
        self.apply_formula = kwargs.get(
            "apply_formula", self.configs["config"]["formula"])
        # table config
        self.table_config = kwargs.get(
            "table_config", self.configs["config"]["table_config"])
        self.apply_table = self.table_config.get(
            "is_table_recog_enable", False)
        self.table_max_time = self.table_config.get(
            "max_time", TABLE_MAX_TIME_VALUE)
        self.table_model_type = self.table_config.get("model", TABLE_MASTER)
=======

        # layout config
        self.layout_config = kwargs.get("layout_config")
        self.layout_model_name = self.layout_config.get("model", MODEL_NAME.DocLayout_YOLO)

        # formula config
        self.formula_config = kwargs.get("formula_config")
        self.mfd_model_name = self.formula_config.get("mfd_model", MODEL_NAME.YOLO_V8_MFD)
        self.mfr_model_name = self.formula_config.get("mfr_model", MODEL_NAME.UniMerNet_v2_Small)
        self.apply_formula = self.formula_config.get("enable", True)

        # table config
        self.table_config = kwargs.get("table_config")
        self.apply_table = self.table_config.get("enable", False)
        self.table_max_time = self.table_config.get("max_time", TABLE_MAX_TIME_VALUE)
        self.table_model_name = self.table_config.get("model", MODEL_NAME.RAPID_TABLE)

        # ocr config
>>>>>>> eadf4ce7c3ac4d502ee626738b72da9b71819c4d
        self.apply_ocr = ocr
        self.lang = kwargs.get("lang", None)

        logger.info(
            "DocAnalysis init, this may take some times, layout_model: {}, apply_formula: {}, apply_ocr: {}, "
            "apply_table: {}, table_model: {}, lang: {}".format(
                self.layout_model_name, self.apply_formula, self.apply_ocr, self.apply_table, self.table_model_name,
                self.lang
            )
        )
        # 初始化解析方案
        self.device = kwargs.get("device", "cpu")
        logger.info("using device: {}".format(self.device))
        models_dir = kwargs.get("models_dir", os.path.join(
            root_dir, "resources", "models"))
        logger.info("using models_dir: {}".format(models_dir))

        atom_model_manager = AtomModelSingleton()

        # 初始化公式识别
        if self.apply_formula:
            # 初始化公式检测模型
            self.mfd_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.MFD,
<<<<<<< HEAD
                mfd_weights=str(os.path.join(
                    models_dir, self.configs["weights"]["mfd"]))
=======
                mfd_weights=str(os.path.join(models_dir, self.configs["weights"][self.mfd_model_name])),
                device=self.device
>>>>>>> eadf4ce7c3ac4d502ee626738b72da9b71819c4d
            )

            # 初始化公式解析模型
<<<<<<< HEAD
            mfr_weight_dir = str(os.path.join(
                models_dir, self.configs["weights"]["mfr"]))
            mfr_cfg_path = str(os.path.join(
                model_config_dir, "UniMERNet", "demo.yaml"))
            # self.mfr_model, mfr_vis_processors = mfr_model_init(mfr_weight_dir, mfr_cfg_path, _device_=self.device)
            # self.mfr_transform = transforms.Compose([mfr_vis_processors, ])
            self.mfr_model, self.mfr_transform = atom_model_manager.get_atom_model(
=======
            mfr_weight_dir = str(os.path.join(models_dir, self.configs["weights"][self.mfr_model_name]))
            mfr_cfg_path = str(os.path.join(model_config_dir, "UniMERNet", "demo.yaml"))
            self.mfr_model = atom_model_manager.get_atom_model(
>>>>>>> eadf4ce7c3ac4d502ee626738b72da9b71819c4d
                atom_model_name=AtomicModel.MFR,
                mfr_weight_dir=mfr_weight_dir,
                mfr_cfg_path=mfr_cfg_path,
                device=self.device
            )

        # 初始化layout模型
<<<<<<< HEAD
        # self.layout_model = Layoutlmv3_Predictor(
        #     str(os.path.join(models_dir, self.configs['weights']['layout'])),
        #     str(os.path.join(model_config_dir, "layoutlmv3", "layoutlmv3_base_inference.yaml")),
        #     device=self.device
        # )
        self.layout_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.Layout,
            layout_weights=str(os.path.join(
                models_dir, self.configs['weights']['layout'])),
            layout_config_file=str(os.path.join(
                model_config_dir, "layoutlmv3", "layoutlmv3_base_inference.yaml")),
            device=self.device
        )
=======
        if self.layout_model_name == MODEL_NAME.LAYOUTLMv3:
            self.layout_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Layout,
                layout_model_name=MODEL_NAME.LAYOUTLMv3,
                layout_weights=str(os.path.join(models_dir, self.configs['weights'][self.layout_model_name])),
                layout_config_file=str(os.path.join(model_config_dir, "layoutlmv3", "layoutlmv3_base_inference.yaml")),
                device=self.device
            )
        elif self.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            self.layout_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Layout,
                layout_model_name=MODEL_NAME.DocLayout_YOLO,
                doclayout_yolo_weights=str(os.path.join(models_dir, self.configs['weights'][self.layout_model_name])),
                device=self.device
            )
>>>>>>> eadf4ce7c3ac4d502ee626738b72da9b71819c4d
        # 初始化ocr
        if self.apply_ocr:
            self.ocr_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.OCR,
                ocr_show_log=show_log,
                det_db_box_thresh=0.3,
                lang=self.lang
            )
        # init table model
        if self.apply_table:
            table_model_dir = self.configs["weights"][self.table_model_name]
            self.table_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.Table,
<<<<<<< HEAD
                table_model_type=self.table_model_type,
                table_model_path=str(os.path.join(
                    models_dir, table_model_dir)),
=======
                table_model_name=self.table_model_name,
                table_model_path=str(os.path.join(models_dir, table_model_dir)),
>>>>>>> eadf4ce7c3ac4d502ee626738b72da9b71819c4d
                table_max_time=self.table_max_time,
                device=self.device
            )

        logger.info('DocAnalysis init done!')

    def __call__(self, image):

        page_start = time.time()

        # layout检测
        layout_start = time.time()
        layout_res = []
        if self.layout_model_name == MODEL_NAME.LAYOUTLMv3:
            # layoutlmv3
            layout_res = self.layout_model(image, ignore_catids=[])
        elif self.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            # doclayout_yolo
            layout_res = self.layout_model.predict(image)
        layout_cost = round(time.time() - layout_start, 2)
        logger.info(f"layout detection time: {layout_cost}")

        pil_img = Image.fromarray(image)

        if self.apply_formula:
            # 公式检测
<<<<<<< HEAD
            mfd_res = self.mfd_model.predict(
                image, imgsz=1888, conf=0.25, iou=0.45, verbose=True)[0]
            for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                new_item = {
                    'category_id': 13 + int(cla.item()),
                    'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    'score': round(float(conf.item()), 2),
                    'latex': '',
                }
                layout_res.append(new_item)
                latex_filling_list.append(new_item)
                bbox_img = get_croped_image(Image.fromarray(image), [
                                            xmin, ymin, xmax, ymax])
                mf_image_list.append(bbox_img)
=======
            mfd_start = time.time()
            mfd_res = self.mfd_model.predict(image)
            logger.info(f"mfd time: {round(time.time() - mfd_start, 2)}")
>>>>>>> eadf4ce7c3ac4d502ee626738b72da9b71819c4d

            # 公式识别
            mfr_start = time.time()
            formula_list = self.mfr_model.predict(mfd_res, image)
            layout_res.extend(formula_list)
            mfr_cost = round(time.time() - mfr_start, 2)
<<<<<<< HEAD
            logger.info(
                f"formula nums: {len(mf_image_list)}, mfr time: {mfr_cost}")
=======
            logger.info(f"formula nums: {len(formula_list)}, mfr time: {mfr_cost}")
>>>>>>> eadf4ce7c3ac4d502ee626738b72da9b71819c4d

        # 清理显存
        clean_vram(self.device, vram_threshold=8)

<<<<<<< HEAD
        #  Unified crop img logic
        def crop_img(input_res, input_pil_img, crop_paste_x=0, crop_paste_y=0):
            crop_xmin, crop_ymin = int(
                input_res['poly'][0]), int(input_res['poly'][1])
            crop_xmax, crop_ymax = int(
                input_res['poly'][4]), int(input_res['poly'][5])
            # Create a white background with an additional width and height of 50
            crop_new_width = crop_xmax - crop_xmin + crop_paste_x * 2
            crop_new_height = crop_ymax - crop_ymin + crop_paste_y * 2
            return_image = Image.new(
                'RGB', (crop_new_width, crop_new_height), 'white')

            # Crop image
            crop_box = (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
            cropped_img = input_pil_img.crop(crop_box)
            return_image.paste(cropped_img, (crop_paste_x, crop_paste_y))
            return_list = [crop_paste_x, crop_paste_y, crop_xmin, crop_ymin,
                           crop_xmax, crop_ymax, crop_new_width, crop_new_height]
            return return_image, return_list

        pil_img = Image.fromarray(image)
=======
        # 从layout_res中获取ocr区域、表格区域、公式区域
        ocr_res_list, table_res_list, single_page_mfdetrec_res = get_res_list_from_layout_res(layout_res)
>>>>>>> eadf4ce7c3ac4d502ee626738b72da9b71819c4d

        # ocr识别
        if self.apply_ocr:
            ocr_start = time.time()
            # Process each area that requires OCR processing
            for res in ocr_res_list:
<<<<<<< HEAD
                new_image, useful_list = crop_img(
                    res, pil_img, crop_paste_x=50, crop_paste_y=50)
                paste_x, paste_y, xmin, ymin, xmax, ymax, new_width, new_height = useful_list
                # Adjust the coordinates of the formula area
                adjusted_mfdetrec_res = []
                for mf_res in single_page_mfdetrec_res:
                    mf_xmin, mf_ymin, mf_xmax, mf_ymax = mf_res["bbox"]
                    # Adjust the coordinates of the formula area to the coordinates relative to the cropping area
                    x0 = mf_xmin - xmin + paste_x
                    y0 = mf_ymin - ymin + paste_y
                    x1 = mf_xmax - xmin + paste_x
                    y1 = mf_ymax - ymin + paste_y
                    # Filter formula blocks outside the graph
                    if any([x1 < 0, y1 < 0]) or any([x0 > new_width, y0 > new_height]):
                        continue
                    else:
                        adjusted_mfdetrec_res.append({
                            "bbox": [x0, y0, x1, y1],
                        })
=======
                new_image, useful_list = crop_img(res, pil_img, crop_paste_x=50, crop_paste_y=50)
                adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(single_page_mfdetrec_res, useful_list)
>>>>>>> eadf4ce7c3ac4d502ee626738b72da9b71819c4d

                # OCR recognition
                new_image = cv2.cvtColor(
                    np.asarray(new_image), cv2.COLOR_RGB2BGR)
                ocr_res = self.ocr_model.ocr(
                    new_image, mfd_res=adjusted_mfdetrec_res)[0]

                # Integration results
                if ocr_res:
                    ocr_result_list = get_ocr_result_list(ocr_res, useful_list)
                    layout_res.extend(ocr_result_list)

            ocr_cost = round(time.time() - ocr_start, 2)
            logger.info(f"ocr time: {ocr_cost}")

        # 表格识别 table recognition
        if self.apply_table:
            table_start = time.time()
            for res in table_res_list:
                new_image, _ = crop_img(res, pil_img)
                single_table_start_time = time.time()
<<<<<<< HEAD
                logger.info(
                    "---------table recognition processing begins-------")
                latex_code = None
=======
>>>>>>> eadf4ce7c3ac4d502ee626738b72da9b71819c4d
                html_code = None
                if self.table_model_name == MODEL_NAME.STRUCT_EQTABLE:
                    with torch.no_grad():
                        table_result = self.table_model.predict(new_image, "html")
                        if len(table_result) > 0:
                            html_code = table_result[0]
                elif self.table_model_name == MODEL_NAME.TABLE_MASTER:
                    html_code = self.table_model.img2html(new_image)
                elif self.table_model_name == MODEL_NAME.RAPID_TABLE:
                    html_code, table_cell_bboxes, elapse = self.table_model.predict(new_image)
                run_time = time.time() - single_table_start_time
<<<<<<< HEAD
                logger.info(
                    f"------table recognition processing ends within {run_time}s-----")
                if run_time > self.table_max_time:
                    logger.warning(
                        f"-------table recognition processing exceeds max time {self.table_max_time}s-------")

                # 判断是否返回正常
                if latex_code:
                    expected_ending = latex_code.strip().endswith('end{tabular}') or latex_code.strip().endswith(
                        'end{table}')
=======
                if run_time > self.table_max_time:
                    logger.warning(f"table recognition processing exceeds max time {self.table_max_time}s")
                # 判断是否返回正常
                if html_code:
                    expected_ending = html_code.strip().endswith('</html>') or html_code.strip().endswith('</table>')
>>>>>>> eadf4ce7c3ac4d502ee626738b72da9b71819c4d
                    if expected_ending:
                        res["html"] = html_code
                    else:
<<<<<<< HEAD
                        logger.warning(
                            f"--------table recognition processing fails--------")
                elif html_code:
                    res["html"] = html_code
                else:
                    logger.warning(
                        f"-------table recognition processing fails-------")
            table_cost = round(time.time() - table_start, 2)
            logger.info(f"table cost: {table_cost}")
=======
                        logger.warning(f"table recognition processing fails, not found expected HTML table end")
                else:
                    logger.warning(f"table recognition processing fails, not get html return")
            logger.info(f"table time: {round(time.time() - table_start, 2)}")

        logger.info(f"-----page total time: {round(time.time() - page_start, 2)}-----")
>>>>>>> eadf4ce7c3ac4d502ee626738b72da9b71819c4d

        return layout_res
