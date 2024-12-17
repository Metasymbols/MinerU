from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor, default_setup

from .backbone import *
from .rcnn_vl import *
from .visualizer import Visualizer


def add_vit_config(cfg):
    """Add config for VIT.

    Args:
        cfg: Configuration node to be modified.
    """
    _C = cfg
    _C.MODEL.VIT = CN()
    _C.MODEL.VIT.NAME = ''
    _C.MODEL.VIT.OUT_FEATURES = ['layer3', 'layer5', 'layer7', 'layer11']
    _C.MODEL.VIT.IMG_SIZE = [224, 224]
    _C.MODEL.VIT.POS_TYPE = 'shared_rel'
    _C.MODEL.VIT.DROP_PATH = 0.
    _C.MODEL.VIT.MODEL_KWARGS = '{}'
    _C.SOLVER.OPTIMIZER = 'ADAMW'
    _C.SOLVER.BACKBONE_MULTIPLIER = 1.0
    _C.AUG = CN()
    _C.AUG.DETR = False
    _C.MODEL.IMAGE_ONLY = True
    _C.PUBLAYNET_DATA_DIR_TRAIN = ''
    _C.PUBLAYNET_DATA_DIR_TEST = ''
    _C.FOOTNOTE_DATA_DIR_TRAIN = ''
    _C.FOOTNOTE_DATA_DIR_VAL = ''
    _C.SCIHUB_DATA_DIR_TRAIN = ''
    _C.SCIHUB_DATA_DIR_TEST = ''
    _C.JIAOCAI_DATA_DIR_TRAIN = ''
    _C.JIAOCAI_DATA_DIR_TEST = ''
    _C.ICDAR_DATA_DIR_TRAIN = ''
    _C.ICDAR_DATA_DIR_TEST = ''
    _C.M6DOC_DATA_DIR_TEST = ''
    _C.DOCSTRUCTBENCH_DATA_DIR_TEST = ''
    _C.DOCSTRUCTBENCHv2_DATA_DIR_TEST = ''
    _C.CACHE_DIR = ''
    _C.MODEL.CONFIG_PATH = ''
    _C.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1


def setup(args, device):
    """Create configs and perform basic setups.

    Args:
        args: Command line arguments.
        device: Device to run the model on.

    Returns:
        cfg: Configured settings.
    """
    cfg = get_cfg()
    add_vit_config(cfg)
    try:
        cfg.merge_from_file(args.config_file)
    except Exception as e:
        print(f'Error loading config file: {e}')
        raise
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.merge_from_list(args.opts)
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        if key not in self.keys():
            return None
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value

    def __setattr__(self, key, value):
        self[key] = value


class Layoutlmv3_Predictor(object):
    def __init__(self, weights, config_file, device):
        layout_args = {
            'config_file': config_file,
            'resume': False,
            'eval_only': False,
            'num_gpus': 1,
            'num_machines': 1,
            'machine_rank': 0,
            'dist_url': 'tcp://127.0.0.1:57823',
            'opts': ['MODEL.WEIGHTS', weights],
        }
        layout_args = DotDict(layout_args)

        cfg = setup(layout_args, device)
        self.mapping = ['title', 'plain text', 'abandon', 'figure', 'figure_caption', 'table', 'table_caption',
                        'table_footnote', 'isolate_formula', 'formula_caption']
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = self.mapping
        self.predictor = DefaultPredictor(cfg)

    def __call__(self, image, ignore_catids=[]):
        # page_layout_result = {
        #     "layout_dets": []
        # }
        layout_dets = []
        outputs = self.predictor(image)
        boxes = outputs['instances'].to(
            'cpu')._fields['pred_boxes'].tensor.tolist()
        labels = outputs['instances'].to(
            'cpu')._fields['pred_classes'].tolist()
        scores = outputs['instances'].to('cpu')._fields['scores'].tolist()
        for bbox_idx in range(len(boxes)):
            if labels[bbox_idx] in ignore_catids:
                continue
            layout_dets.append({
                'category_id': labels[bbox_idx],
                'poly': [
                    boxes[bbox_idx][0], boxes[bbox_idx][1],
                    boxes[bbox_idx][2], boxes[bbox_idx][1],
                    boxes[bbox_idx][2], boxes[bbox_idx][3],
                    boxes[bbox_idx][0], boxes[bbox_idx][3],
                ],
                'score': scores[bbox_idx]
            })
        return layout_dets
