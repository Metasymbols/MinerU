import copy
import math
import os
import re
import statistics
import time
from typing import List

import cv2
import fitz
import torch
import numpy as np
from loguru import logger

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.data.dataset import Dataset, PageableData
from magic_pdf.libs.boxbase import calculate_overlap_area_in_bbox1_area_ratio, __is_overlaps_y_exceeds_threshold
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.libs.config_reader import get_local_layoutreader_model_dir, get_llm_aided_config, get_device
from magic_pdf.libs.convert_utils import dict_to_list
from magic_pdf.libs.hash_utils import compute_md5
from magic_pdf.libs.pdf_image_tools import cut_image_to_pil_image
from magic_pdf.model.magic_model import MagicModel
from magic_pdf.post_proc.llm_aided import llm_aided_formula, llm_aided_text, llm_aided_title

try:
    import torchtext

    if torchtext.__version__ >= '0.18.0':
        torchtext.disable_torchtext_deprecation_warning()
except ImportError:
    pass

from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
from magic_pdf.post_proc.para_split_v3 import para_split
from magic_pdf.pre_proc.construct_page_dict import ocr_construct_page_component_v2
from magic_pdf.pre_proc.cut_image import ocr_cut_image_and_table
from magic_pdf.pre_proc.ocr_detect_all_bboxes import ocr_prepare_bboxes_for_layout_split_v2
from magic_pdf.pre_proc.ocr_dict_merge import fill_spans_in_blocks, fix_block_spans_v2, fix_discarded_block
from magic_pdf.pre_proc.ocr_span_list_modify import get_qa_need_list_v2, remove_overlaps_low_confidence_spans, \
    remove_overlaps_min_spans, check_chars_is_overlap_in_span

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新

class ModelSingleton:
    _instance = None
    _models = {}
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models = {}
        return cls._instance

    def get_model(self, model_name: str):
        if model_name not in self._models:
            with self._lock:
                if model_name not in self._models:
                    self._models[model_name] = model_init(model_name=model_name)
        return self._models[model_name]

    def clear_models(self):
        """清理所有已加载的模型"""
        with self._lock:
            self._models.clear()

def __replace_STX_ETX(text_str: str) -> str:
    """替换使用pymupdf提取时出现乱码的\u0002和\u0003字符，这些字符原本是引号。
    注意：此问题目前仅在英文文本中观察到，在中文文本中尚未发现。

    Args:
        text_str (str): 原始文本

    Returns:
        str: 替换后的文本
    """
    if not text_str:
        return text_str
    return text_str.replace('\u0002', "'").replace('\u0003', "'")

def __replace_0xfffd(text_str: str) -> str:
    """替换使用pymupdf提取时出现乱码的\ufffd字符。

    Args:
        text_str (str): 原始文本

    Returns:
        str: 替换后的文本
    """
    if not text_str:
        return text_str
    return text_str.replace('\ufffd', " ")

def __replace_ligatures(text: str) -> str:
    """拆分连写字符。

    Args:
        text (str): 包含连写字符的文本

    Returns:
        str: 拆分后的文本
    """
    ligatures = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬅ': 'ft', 'ﬆ': 'st'
    }
    return re.sub('|'.join(map(re.escape, ligatures.keys())), lambda m: ligatures[m.group()], text)
def chars_to_content(span: dict) -> None:
    """将字符列表转换为文本内容。
    此函数处理span中的字符列表，将其转换为连续的文本内容。主要步骤包括：
    1. 检查字符列表的有效性
    2. 按字符的中心点x坐标排序
    3. 根据字符间距自动插入空格
    4. 处理特殊字符（连字和乱码字符）

    处理规则：
    - 忽略空的字符列表
    - 跳过存在重叠字符的span
    - 当相邻字符间距超过平均字符宽度的25%时插入空格
    - 自动处理连字符和乱码字符

    Args:
        span (dict): 包含字符列表的span字典，必须包含'chars'键
                    每个字符必须包含'bbox'和'c'属性

    Returns:
        None: 直接修改输入的span字典，添加'content'键并删除'chars'键

    Note:
        - bbox格式：[x0, y0, x1, y1]
        - 字符间距判断基于字符的边界框坐标
    """
    # 输入验证
    if not isinstance(span, dict) or 'chars' not in span:
        return

    # 空字符列表处理
    if len(span['chars']) == 0:
        span['content'] = ''
        return

    # 重叠字符检查
    if check_chars_is_overlap_in_span(span['chars']):
        span['content'] = ''
        return

    # 按字符中心点x坐标排序
    span['chars'] = sorted(span['chars'], key=lambda x: (x['bbox'][0] + x['bbox'][2]) / 2)

    # 计算平均字符宽度
    char_widths = [char['bbox'][2] - char['bbox'][0] for char in span['chars']]
    char_avg_width = sum(char_widths) / len(span['chars'])

    # 构建文本内容
    content = []
    for i, char in enumerate(span['chars']):
        content.append(char['c'])
        
        # 根据字符间距插入空格
        if i < len(span['chars']) - 1:
            next_char = span['chars'][i + 1]
            char_gap = next_char['bbox'][0] - char['bbox'][2]
            if (char_gap > char_avg_width * 0.25 and 
                char['c'] != ' ' and next_char['c'] != ' '):
                content.append(' ')

    # 处理特殊字符
    content = ''.join(content)
    content = __replace_ligatures(content)  # 处理连字
    span['content'] = __replace_0xfffd(content)  # 处理乱码字符

    # 清理临时数据
    del span['chars']

LINE_STOP_FLAG = ('.', '!', '?', '。', '！', '？', ')', '）', '"', '”', ':', '：', ';', '；', ']', '】', '}', '}', '>', '》', '、', ',', '，', '-', '—', '–',)
LINE_START_FLAG = ('(', '（', '"', '“', '【', '{', '《', '<', '「', '『', '【', '[',)

def fill_char_in_spans(spans, all_chars):

    # 简单从上到下排一下序
    spans = sorted(spans, key=lambda x: x['bbox'][1])

    for char in all_chars:
        # 跳过非法bbox的char
        # x1, y1, x2, y2 = char['bbox']
        # if abs(x1 - x2) <= 0.01 or abs(y1 - y2) <= 0.01:
        #     continue

        for span in spans:
            if calculate_char_in_span(char['bbox'], span['bbox'], char['c']):
                span['chars'].append(char)
                break

    need_ocr_spans = []
    for span in spans:
        chars_to_content(span)
        # 有的span中虽然没有字但有一两个空的占位符，用宽高和content长度过滤
        if len(span['content']) * span['height'] < span['width'] * 0.5:
            # logger.info(f"maybe empty span: {len(span['content'])}, {span['height']}, {span['width']}")
            need_ocr_spans.append(span)
        del span['height'], span['width']
    return need_ocr_spans

# 使用鲁棒性更强的中心点坐标判断
def calculate_char_in_span(char_bbox, span_bbox, char, span_height_radio=0.33):
    """判断字符是否在span区域内。

    Args:
        char_bbox: 字符的边界框坐标
        span_bbox: span的边界框坐标
        char: 字符内容
        span_height_radio: 字符中轴与span中轴的高度差阈值比例

    Returns:
        bool: 字符是否在span区域内
    """
    char_center_x = (char_bbox[0] + char_bbox[2]) / 2
    char_center_y = (char_bbox[1] + char_bbox[3]) / 2
    span_center_y = (span_bbox[1] + span_bbox[3]) / 2
    span_height = span_bbox[3] - span_bbox[1]

    # 基本位置判断
    if (span_bbox[0] < char_center_x < span_bbox[2] and
        span_bbox[1] < char_center_y < span_bbox[3] and
        abs(char_center_y - span_center_y) < span_height * span_height_radio):
        return True

    # 特殊字符处理
    if char in LINE_STOP_FLAG:
        # 结尾符号特殊处理
        return ((span_bbox[2] - span_height) < char_bbox[0] < span_bbox[2] and
                char_center_x > span_bbox[0] and
                span_bbox[1] < char_center_y < span_bbox[3] and
                abs(char_center_y - span_center_y) < span_height * span_height_radio)
    elif char in LINE_START_FLAG:
        # 开始符号特殊处理
        return (span_bbox[0] < char_bbox[2] < (span_bbox[0] + span_height) and
                char_center_x < span_bbox[2] and
                span_bbox[1] < char_center_y < span_bbox[3] and
                abs(char_center_y - span_center_y) < span_height * span_height_radio)

    return False

def remove_tilted_line(text_blocks):
    """移除倾斜的文本行。

    Args:
        text_blocks (list): 文本块列表
    """
    for block in text_blocks:
        remove_lines = []
        for line in block['lines']:
            cosine, sine = line['dir']
            angle_radians = math.atan2(sine, cosine)
            angle_degrees = math.degrees(angle_radians)
            if 2 < abs(angle_degrees) < 88:
                remove_lines.append(line)
        for line in remove_lines:
            block['lines'].remove(line)

def calculate_contrast(img, img_mode) -> float:
    """计算给定图像的对比度。
    此函数通过计算图像灰度值的标准差与均值的比值来衡量图像的对比度。
    对比度值越大，表示图像中明暗差异越大；对比度值越小，表示图像越趋于平淡。

    计算步骤：
    1. 将输入图像转换为灰度图
    2. 计算灰度图的均值和标准差
    3. 计算对比度 = 标准差 / (均值 + ε)
    其中ε是一个小常数，用于避免除零错误

    Args:
        img (numpy.ndarray): 输入图像，必须是有效的numpy数组
        img_mode (str): 图像的色彩通道模式，必须是'rgb'或'bgr'

    Returns:
        float: 图像的对比度值，保留两位小数

    Raises:
        ValueError: 当img_mode不是'rgb'或'bgr'时抛出
        TypeError: 当输入图像不是numpy数组时抛出
        cv2.error: 当图像转换失败时抛出
    """
    # 输入验证
    if not isinstance(img, np.ndarray):
        raise TypeError("Input image must be a numpy array")
    
    if img.size == 0:
        raise ValueError("Input image is empty")

    # 转换为灰度图
    try:
        if img_mode == 'rgb':
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img_mode == 'bgr':
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Invalid image mode. Please provide 'rgb' or 'bgr'.")
    except cv2.error as e:
        raise cv2.error(f"Failed to convert image to grayscale: {str(e)}")

    # 计算对比度
    mean_value = np.mean(gray_img)
    std_dev = np.std(gray_img)
    epsilon = 1e-6  # 避免除零错误的小常数
    contrast = std_dev / (mean_value + epsilon)
    
    return round(contrast, 2)

def txt_spans_extract_v2(pdf_page, spans, all_bboxes, all_discarded_blocks, lang):
    # cid用0xfffd表示，连字符拆开
    # text_blocks_raw = pdf_page.get_text('rawdict', flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP)['blocks']

    # cid用0xfffd表示，连字符不拆开
    #text_blocks_raw = pdf_page.get_text('rawdict', flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP)['blocks']

    # 自定义flags出现较多0xfffd，可能是pymupdf可以自行处理内置字典的pdf，不再使用
    text_blocks_raw = pdf_page.get_text('rawdict', flags=fitz.TEXTFLAGS_TEXT)['blocks']
    # text_blocks = pdf_page.get_text('dict', flags=fitz.TEXTFLAGS_TEXT)['blocks']

    # 移除所有角度不为0或90的line
    remove_tilted_line(text_blocks_raw)

    all_pymu_chars = []
    for block in text_blocks_raw:
        for line in block['lines']:
            cosine, sine = line['dir']
            if abs(cosine) < 0.9 or abs(sine) > 0.1:
                continue
            for span in line['spans']:
                all_pymu_chars.extend(span['chars'])

    # 计算所有sapn的高度的中位数
    span_height_list = []
    for span in spans:
        if span['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table]:
            continue
        span_height = span['bbox'][3] - span['bbox'][1]
        span['height'] = span_height
        span['width'] = span['bbox'][2] - span['bbox'][0]
        span_height_list.append(span_height)
    if len(span_height_list) == 0:
        return spans
    else:
        median_span_height = statistics.median(span_height_list)

    useful_spans = []
    unuseful_spans = []
    # 纵向span的两个特征：1. 高度超过多个line 2. 高宽比超过某个值
    vertical_spans = []
    for span in spans:
        if span['type'] in [ContentType.InterlineEquation, ContentType.Image, ContentType.Table]:
            continue
        for block in all_bboxes + all_discarded_blocks:
            if block[7] in [BlockType.ImageBody, BlockType.TableBody, BlockType.InterlineEquation]:
                continue
            if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block[0:4]) > 0.5:
                if span['height'] > median_span_height * 3 and span['height'] > span['width'] * 3:
                    vertical_spans.append(span)
                elif block in all_bboxes:
                    useful_spans.append(span)
                else:
                    unuseful_spans.append(span)

                break

    """垂直的span框直接用pymu的line进行填充"""
    if len(vertical_spans) > 0:
        text_blocks = pdf_page.get_text('dict', flags=fitz.TEXTFLAGS_TEXT)['blocks']
        all_pymu_lines = []
        for block in text_blocks:
            for line in block['lines']:
                all_pymu_lines.append(line)

        for pymu_line in all_pymu_lines:
            for span in vertical_spans:
                if calculate_overlap_area_in_bbox1_area_ratio(pymu_line['bbox'], span['bbox']) > 0.5:
                    for pymu_span in pymu_line['spans']:
                        span['content'] += pymu_span['text']
                    break

        for span in vertical_spans:
            if len(span['content']) == 0:
                spans.remove(span)

    """水平的span框如果没有char则用ocr进行填充"""
    new_spans = []

    for span in useful_spans + unuseful_spans:
        if span['type'] in [ContentType.Text]:
            span['chars'] = []
            new_spans.append(span)

    need_ocr_spans = fill_char_in_spans(new_spans, all_pymu_chars)

    if len(need_ocr_spans) > 0:

        # 初始化ocr模型
        atom_model_manager = AtomModelSingleton()
        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            ocr_show_log=False,
            det_db_box_thresh=0.3,
            lang=lang
        )

        for span in need_ocr_spans:
            # 对span的bbox截图再ocr
            span_img = cut_image_to_pil_image(span['bbox'], pdf_page, mode='cv2')

            # 计算span的对比度，低于0.20的span不进行ocr
            if calculate_contrast(span_img, img_mode='bgr') <= 0.20:
                spans.remove(span)
                continue

            ocr_res = ocr_model.ocr(span_img, det=False)
            if ocr_res and len(ocr_res) > 0:
                if len(ocr_res[0]) > 0:
                    ocr_text, ocr_score = ocr_res[0][0]
                    # logger.info(f"ocr_text: {ocr_text}, ocr_score: {ocr_score}")
                    if ocr_score > 0.5 and len(ocr_text) > 0:
                        span['content'] = ocr_text
                        span['score'] = ocr_score
                    else:
                        spans.remove(span)

    return spans

def model_init(model_name: str):
    """初始化模型。

    Args:
        model_name (str): 模型名称

    Returns:
        model: 初始化后的模型实例

    Raises:
        SystemExit: 当模型名称不支持时抛出异常
    """
    from transformers import LayoutLMv3ForTokenClassification
    device = torch.device(get_device())

    if model_name == 'layoutreader':
        # 检测modelscope的缓存目录是否存在
        layoutreader_model_dir = get_local_layoutreader_model_dir()
        if os.path.exists(layoutreader_model_dir):
            model = LayoutLMv3ForTokenClassification.from_pretrained(
                layoutreader_model_dir
            )
        else:
            logger.warning(
                'local layoutreader model not exists, use online model from huggingface'
            )
            model = LayoutLMv3ForTokenClassification.from_pretrained(
                'hantian/layoutreader'
            )
        model.to(device).eval()
    else:
        logger.error('model name not allow')
        exit(1)
    return model

def do_predict(boxes: List[List[int]], model) -> List[int]:
    """使用模型进行预测。

    Args:
        boxes (List[List[int]]): 边界框列表
        model: 预测模型

    Returns:
        List[int]: 预测结果列表
    """
    from magic_pdf.model.sub_modules.reading_oreder.layoutreader.helpers import (
        boxes2inputs, parse_logits, prepare_inputs)

    inputs = boxes2inputs(boxes)
    inputs = prepare_inputs(inputs, model)
    logits = model(**inputs).logits.cpu().squeeze(0)
    return parse_logits(logits, len(boxes))

def cal_block_index(fix_blocks, sorted_bboxes):
    """计算块的索引顺序。
    此函数根据不同的排序策略（layoutreader或xycut）为文档块和其中的行分配索引。
    同时处理特殊块类型（如图片、表格、标题等）的行结构。

    排序策略说明：
    1. layoutreader排序：使用预先排序的边界框列表
       - 空块直接使用边界框索引
       - 有行的块使用行索引的中位数作为块索引
    2. xycut排序：使用递归的XY切分算法
       - 先对块进行排序
       - 再按顺序为每一行分配连续的索引

    Args:
        fix_blocks (list): 需要计算索引的块列表，每个块包含bbox和lines信息
        sorted_bboxes (list): 已排序的边界框列表，如果为None则使用xycut排序

    Returns:
        list: 添加了索引信息的块列表，每个块和其中的行都有对应的index
    """
    def _process_special_block(block):
        """处理特殊类型块的行结构"""
        if block['type'] in [BlockType.ImageBody, BlockType.TableBody, BlockType.Title, BlockType.InterlineEquation]:
            if 'real_lines' in block:
                block['virtual_lines'] = copy.deepcopy(block['lines'])
                block['lines'] = copy.deepcopy(block['real_lines'])
                del block['real_lines']

    if sorted_bboxes is not None:
        # 使用layoutreader排序策略
        for block in fix_blocks:
            if len(block['lines']) == 0:
                # 空块直接使用边界框索引
                block['index'] = sorted_bboxes.index(block['bbox'])
            else:
                # 使用行索引的中位数作为块索引
                line_indices = []
                for line in block['lines']:
                    line['index'] = sorted_bboxes.index(line['bbox'])
                    line_indices.append(line['index'])
                block['index'] = statistics.median(line_indices)
            
            _process_special_block(block)
    else:
        # 使用xycut排序策略
        import numpy as np
        from magic_pdf.model.sub_modules.reading_oreder.layoutreader.xycut import recursive_xy_cut

        # 准备块的边界框
        block_bboxes = []
        for block in fix_blocks:
            # 确保坐标非负
            block['bbox'] = [max(0, x) for x in block['bbox']]
            block_bboxes.append(block['bbox'])
            _process_special_block(block)

        # 使用xycut算法排序
        random_boxes = np.array(block_bboxes)
        np.random.shuffle(random_boxes)  # 随机打乱以避免初始顺序的影响
        res = []
        recursive_xy_cut(np.asarray(random_boxes).astype(int), np.arange(len(block_bboxes)), res)
        sorted_boxes = random_boxes[np.array(res)].tolist()

        # 分配块索引
        for block in fix_blocks:
            block['index'] = sorted_boxes.index(block['bbox'])

        # 为行分配连续的索引
        sorted_blocks = sorted(fix_blocks, key=lambda b: b['index'])
        line_index = 1
        for block in sorted_blocks:
            for line in block['lines']:
                line['index'] = line_index
                line_index += 1

    return fix_blocks

def insert_lines_into_block(block_bbox, line_height, page_w, page_h):
    """根据块的大小和页面尺寸插入虚拟行。
    此函数根据块的尺寸特征（高度、宽度、长宽比）和页面布局特征（单列、双列、三列等）
    来确定最合适的分行策略，从而生成虚拟行的边界框列表。

    分行策略说明：
    1. 对于高度小于两倍行高的块，不进行分行
    2. 对于可能的双列结构（高度>页高1/4且1/4页宽<宽度<1/2页宽），按行高精确分行
    3. 对于宽块（宽度>0.4页宽），固定分3行
    4. 对于中等宽度块（宽度>0.25页宽），可能是三列结构，按行高精确分行
    5. 对于窄块，根据长宽比决定是否分行：
       - 长宽比>1.2的细长块不分行
       - 其他情况分成两行

    Args:
        block_bbox (tuple): 块的边界框坐标 (x0, y0, x1, y1)
        line_height (float): 行高
        page_w (float): 页面宽度
        page_h (float): 页面高度

    Returns:
        list: 插入的虚拟行列表，每个元素为[x0, y0, x1, y1]格式的边界框坐标
    """
    x0, y0, x1, y1 = block_bbox
    block_height = y1 - y0
    block_width = x1 - x0

    # 情况1：块高度小于两倍行高，不分行
    if block_height <= line_height * 2:
        return [[x0, y0, x1, y1]]

    # 情况2：可能的双列结构
    if block_height > page_h * 0.25 and page_w * 0.5 > block_width > page_w * 0.25:
        lines = int(block_height / line_height) + 1
    else:
        # 情况3：宽块处理
        if block_width > page_w * 0.4:
            lines = 3
            line_height = block_height / lines
        # 情况4：中等宽度块处理
        elif block_width > page_w * 0.25:
            lines = int(block_height / line_height) + 1
        # 情况5：窄块处理
        else:
            # 检查长宽比
            if block_height / block_width > 1.2:
                return [[x0, y0, x1, y1]]  # 细长块不分行
            else:
                lines = 2  # 其他情况分两行
                line_height = block_height / lines

    # 根据计算出的行数生成虚拟行
    lines_positions = []
    for i in range(lines):
        line_y0 = y0 + i * line_height
        line_y1 = line_y0 + line_height
        lines_positions.append([x0, line_y0, x1, line_y1])

    return lines_positions

def sort_lines_by_model(fix_blocks, page_w, page_h, line_height):
    page_line_list = []

    def add_lines_to_block(b):
        line_bboxes = insert_lines_into_block(b['bbox'], line_height, page_w, page_h)
        b['lines'] = []
        for line_bbox in line_bboxes:
            b['lines'].append({'bbox': line_bbox, 'spans': []})
        page_line_list.extend(line_bboxes)

    for block in fix_blocks:
        if block['type'] in [
            BlockType.Text, BlockType.Title,
            BlockType.ImageCaption, BlockType.ImageFootnote,
            BlockType.TableCaption, BlockType.TableFootnote
        ]:
            if len(block['lines']) == 0:
                add_lines_to_block(block)
            elif block['type'] in [BlockType.Title] and len(block['lines']) == 1 and (block['bbox'][3] - block['bbox'][1]) > line_height * 2:
                block['real_lines'] = copy.deepcopy(block['lines'])
                add_lines_to_block(block)
            else:
                for line in block['lines']:
                    bbox = line['bbox']
                    page_line_list.append(bbox)
        elif block['type'] in [BlockType.ImageBody, BlockType.TableBody, BlockType.InterlineEquation]:
            block['real_lines'] = copy.deepcopy(block['lines'])
            add_lines_to_block(block)

    if len(page_line_list) > 200:  # layoutreader最高支持512line
        return None

    # 使用layoutreader排序
    x_scale = 1000.0 / page_w
    y_scale = 1000.0 / page_h
    boxes = []
    # logger.info(f"Scale: {x_scale}, {y_scale}, Boxes len: {len(page_line_list)}")
    for left, top, right, bottom in page_line_list:
        if left < 0:
            logger.warning(
                f'left < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            left = 0
        if right > page_w:
            logger.warning(
                f'right > page_w, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            right = page_w
        if top < 0:
            logger.warning(
                f'top < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            top = 0
        if bottom > page_h:
            logger.warning(
                f'bottom > page_h, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            bottom = page_h

        left = round(left * x_scale)
        top = round(top * y_scale)
        right = round(right * x_scale)
        bottom = round(bottom * y_scale)
        assert (
            1000 >= right >= left >= 0 and 1000 >= bottom >= top >= 0
        ), f'Invalid box. right: {right}, left: {left}, bottom: {bottom}, top: {top}'  # noqa: E126, E121
        boxes.append([left, top, right, bottom])
    model_manager = ModelSingleton()
    model = model_manager.get_model('layoutreader')
    with torch.no_grad():
        orders = do_predict(boxes, model)
    sorted_bboxes = [page_line_list[i] for i in orders]

    return sorted_bboxes

def get_line_height(blocks):
    page_line_height_list = []
    for block in blocks:
        if block['type'] in [
            BlockType.Text, BlockType.Title,
            BlockType.ImageCaption, BlockType.ImageFootnote,
            BlockType.TableCaption, BlockType.TableFootnote
        ]:
            for line in block['lines']:
                bbox = line['bbox']
                page_line_height_list.append(int(bbox[3] - bbox[1]))
    if len(page_line_height_list) > 0:
        return statistics.median(page_line_height_list)
    else:
        return 10

def process_groups(groups, body_key, caption_key, footnote_key):
    body_blocks = []
    caption_blocks = []
    footnote_blocks = []
    for i, group in enumerate(groups):
        group[body_key]['group_id'] = i
        body_blocks.append(group[body_key])
        for caption_block in group[caption_key]:
            caption_block['group_id'] = i
            caption_blocks.append(caption_block)
        for footnote_block in group[footnote_key]:
            footnote_block['group_id'] = i
            footnote_blocks.append(footnote_block)
    return body_blocks, caption_blocks, footnote_blocks

def process_block_list(blocks, body_type, block_type):
    indices = [block['index'] for block in blocks]
    median_index = statistics.median(indices)

    body_bbox = next((block['bbox'] for block in blocks if block.get('type') == body_type), [])

    return {
        'type': block_type,
        'bbox': body_bbox,
        'blocks': blocks,
        'index': median_index,
    }

def revert_group_blocks(blocks):
    image_groups = {}
    table_groups = {}
    new_blocks = []
    for block in blocks:
        if block['type'] in [BlockType.ImageBody, BlockType.ImageCaption, BlockType.ImageFootnote]:
            group_id = block['group_id']
            if group_id not in image_groups:
                image_groups[group_id] = []
            image_groups[group_id].append(block)
        elif block['type'] in [BlockType.TableBody, BlockType.TableCaption, BlockType.TableFootnote]:
            group_id = block['group_id']
            if group_id not in table_groups:
                table_groups[group_id] = []
            table_groups[group_id].append(block)
        else:
            new_blocks.append(block)

    for group_id, blocks in image_groups.items():
        new_blocks.append(process_block_list(blocks, BlockType.ImageBody, BlockType.Image))

    for group_id, blocks in table_groups.items():
        new_blocks.append(process_block_list(blocks, BlockType.TableBody, BlockType.Table))

    return new_blocks

def remove_outside_spans(spans, all_bboxes, all_discarded_blocks):
    def get_block_bboxes(blocks, block_type_list):
        return [block[0:4] for block in blocks if block[7] in block_type_list]

    image_bboxes = get_block_bboxes(all_bboxes, [BlockType.ImageBody])
    table_bboxes = get_block_bboxes(all_bboxes, [BlockType.TableBody])
    other_block_type = []
    for block_type in BlockType.__dict__.values():
        if not isinstance(block_type, str):
            continue
        if block_type not in [BlockType.ImageBody, BlockType.TableBody]:
            other_block_type.append(block_type)
    other_block_bboxes = get_block_bboxes(all_bboxes, other_block_type)
    discarded_block_bboxes = get_block_bboxes(all_discarded_blocks, [BlockType.Discarded])

    new_spans = []

    for span in spans:
        span_bbox = span['bbox']
        span_type = span['type']

        if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.4 for block_bbox in
               discarded_block_bboxes):
            new_spans.append(span)
            continue

        if span_type == ContentType.Image:
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in
                   image_bboxes):
                new_spans.append(span)
        elif span_type == ContentType.Table:
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in
                   table_bboxes):
                new_spans.append(span)
        else:
            if any(calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.5 for block_bbox in
                   other_block_bboxes):
                new_spans.append(span)

    return new_spans

def parse_page_core(
    page_doc: PageableData, magic_model, page_id, pdf_bytes_md5, imageWriter, parse_mode, lang
):
    """处理PDF页面的核心函数。

    Args:
        page_doc (PageableData): 页面数据对象
        magic_model: 模型对象
        page_id (int): 页面ID
        pdf_bytes_md5 (str): PDF文件的MD5值
        imageWriter: 图像写入器
        parse_mode (str): 解析模式（TXT或OCR）
        lang (str): 语言代码

    Returns:
        dict: 包含页面解析结果的字典
    """
    need_drop = False
    drop_reason = []

    # 1. 从magic_model获取页面基本信息和区块
    page_w, page_h = magic_model.get_page_size(page_id)
    img_groups = magic_model.get_imgs_v2(page_id)
    table_groups = magic_model.get_tables_v2(page_id)
    discarded_blocks = magic_model.get_discarded(page_id)
    text_blocks = magic_model.get_text_blocks(page_id)
    title_blocks = magic_model.get_title_blocks(page_id)
    inline_equations, interline_equations, interline_equation_blocks = magic_model.get_equations(page_id)

    # 2. 处理图片和表格区块
    img_body_blocks, img_caption_blocks, img_footnote_blocks = process_groups(
        img_groups, 'image_body', 'image_caption_list', 'image_footnote_list'
    )

    table_body_blocks, table_caption_blocks, table_footnote_blocks = process_groups(
        table_groups, 'table_body', 'table_caption_list', 'table_footnote_list'
    )

    def merge_title_blocks(blocks, page_w, x_distance_threshold=None):
        """合并同一行的标题块。

        Args:
            blocks (list): 包含标题块的列表
            page_w (float): 页面宽度
            x_distance_threshold (float, optional): 水平距离阈值，默认为页面宽度的10%

        Returns:
            None: 直接修改输入的blocks列表
        """
        if x_distance_threshold is None:
            x_distance_threshold = 0.1 * page_w

    def merge_two_bbox(b1, b2):
        x_min = min(b1['bbox'][0], b2['bbox'][0])
        y_min = min(b1['bbox'][1], b2['bbox'][1])
        x_max = max(b1['bbox'][2], b2['bbox'][2])
        y_max = max(b1['bbox'][3], b2['bbox'][3])
        return x_min, y_min, x_max, y_max

    def merge_two_blocks(b1, b2):
        # 合并两个标题块的边界框
        b1['bbox'] = merge_two_bbox(b1, b2)

        # 合并两个标题块的文本内容
        line1 = b1['lines'][0]
        line2 = b2['lines'][0]
        line1['bbox'] = merge_two_bbox(line1, line2)
        line1['spans'].extend(line2['spans'])

        return b1, b2

    # 按 y 轴重叠度聚集标题块
    y_overlapping_blocks = []
    title_bs = [b for b in blocks if b['type'] == BlockType.Title]
    while title_bs:
        block1 = title_bs.pop(0)
        current_row = [block1]
        to_remove = []
        for block2 in title_bs:
            if (
                __is_overlaps_y_exceeds_threshold(block1['bbox'], block2['bbox'], 0.9)
                and len(block1['lines']) == 1
                and len(block2['lines']) == 1
            ):
                current_row.append(block2)
                to_remove.append(block2)
        for b in to_remove:
            title_bs.remove(b)
        y_overlapping_blocks.append(current_row)

    # 按x轴坐标排序并合并标题块
    to_remove_blocks = []
    for row in y_overlapping_blocks:
        if len(row) == 1:
            continue

        # 按x轴坐标排序
        row.sort(key=lambda x: x['bbox'][0])

        merged_block = row[0]
        for i in range(1, len(row)):
            left_block = merged_block
            right_block = row[i]

            left_height = left_block['bbox'][3] - left_block['bbox'][1]
            right_height = right_block['bbox'][3] - right_block['bbox'][1]

            if (
                right_block['bbox'][0] - left_block['bbox'][2] < x_distance_threshold
                and left_height * 0.95 < right_height < left_height * 1.05
            ):
                merged_block, to_remove_block = merge_two_blocks(merged_block, right_block)
                to_remove_blocks.append(to_remove_block)
            else:
                merged_block = right_block

    for b in to_remove_blocks:
        blocks.remove(b)

    # 3. 整理所有区块的边界框
    # 注意：interline_equation_blocks参数不够准确，使用interline_equations替代
    interline_equation_blocks = []
    all_bboxes, all_discarded_blocks = ocr_prepare_bboxes_for_layout_split_v2(
        img_body_blocks, img_caption_blocks, img_footnote_blocks,
        table_body_blocks, table_caption_blocks, table_footnote_blocks,
        discarded_blocks,
        text_blocks,
        title_blocks,
        interline_equations,  # 使用interline_equations替代interline_equation_blocks
        page_w,
        page_h,
    )

    # 4. 处理spans信息
    # 4.1 获取并过滤spans
    spans = magic_model.get_all_spans(page_id)
    spans = remove_outside_spans(spans, all_bboxes, all_discarded_blocks)  # 过滤图片和表格的span，同时处理水印

    # 4.2 删除重叠的spans
    spans, _ = remove_overlaps_low_confidence_spans(spans)  # 删除重叠spans中置信度较低的
    spans, _ = remove_overlaps_min_spans(spans)  # 删除重叠spans中较小的

    # 5. 根据解析模式处理spans
    if parse_mode == SupportedPdfParseMethod.TXT:
        # 使用混合OCR方案处理文本
        spans = txt_spans_extract_v2(page_doc, spans, all_bboxes, all_discarded_blocks, lang)
    elif parse_mode == SupportedPdfParseMethod.OCR:
        pass
    else:
        raise Exception('parse_mode must be txt or ocr')

    # 6. 处理discarded_blocks
    discarded_block_with_spans, spans = fill_spans_in_blocks(
        all_discarded_blocks, spans, 0.4
    )
    fix_discarded_blocks = fix_discarded_block(discarded_block_with_spans)

    # 7. 处理空页面
    if len(all_bboxes) == 0:
        logger.warning(f'skip this page, not found useful bbox, page_id: {page_id}')
        return ocr_construct_page_component_v2(
            [], [], page_id, page_w, page_h, [], [], [],
            interline_equations, fix_discarded_blocks, need_drop, drop_reason
        )

    # 8. 处理图像和表格
    spans = ocr_cut_image_and_table(
        spans, page_doc, page_id, pdf_bytes_md5, imageWriter
    )

    # 9. 处理blocks
    # 9.1 填充和修复blocks
    block_with_spans, spans = fill_spans_in_blocks(all_bboxes, spans, 0.5)
    fix_blocks = fix_block_spans_v2(block_with_spans)

    # 9.2 合并标题块
    merge_title_blocks(fix_blocks, page_w)

    # 9.3 计算行高和排序
    line_height = get_line_height(fix_blocks)
    sorted_bboxes = sort_lines_by_model(fix_blocks, page_w, page_h, line_height)
    fix_blocks = cal_block_index(fix_blocks, sorted_bboxes)

    # 9.4 还原和重排blocks
    fix_blocks = revert_group_blocks(fix_blocks)
    sorted_blocks = sorted(fix_blocks, key=lambda b: b['index'])

    # 9.5 对图片和表格块内部排序
    for block in sorted_blocks:
        if block['type'] in [BlockType.Image, BlockType.Table]:
            block['blocks'] = sorted(block['blocks'], key=lambda b: b['index'])

    # 10. 构造返回结果
    images, tables, interline_equations = get_qa_need_list_v2(sorted_blocks)
    page_info = ocr_construct_page_component_v2(
        sorted_blocks, [], page_id, page_w, page_h, [],
        images, tables, interline_equations, fix_discarded_blocks,
        need_drop, drop_reason
    )
    return page_info


def process_llm_aided(pdf_info_dict, llm_aided_config):
    """处理LLM辅助优化。

    Args:
        pdf_info_dict (dict): PDF文档信息字典
        llm_aided_config (dict): LLM优化配置信息
    """
    if llm_aided_config is None:
        return

    # 公式优化
    formula_aided_config = llm_aided_config.get('formula_aided', None)
    if formula_aided_config and formula_aided_config.get('enable', False):
        llm_aided_formula_start_time = time.time()
        llm_aided_formula(pdf_info_dict, formula_aided_config)
        logger.info(f'llm aided formula time: {round(time.time() - llm_aided_formula_start_time, 2)}')

    # 文本优化
    text_aided_config = llm_aided_config.get('text_aided', None)
    if text_aided_config and text_aided_config.get('enable', False):
        llm_aided_text_start_time = time.time()
        llm_aided_text(pdf_info_dict, text_aided_config)
        logger.info(f'llm aided text time: {round(time.time() - llm_aided_text_start_time, 2)}')

    # 标题优化
    title_aided_config = llm_aided_config.get('title_aided', None)
    if title_aided_config and title_aided_config.get('enable', False):
        llm_aided_title_start_time = time.time()
        llm_aided_title(pdf_info_dict, title_aided_config)
        logger.info(f'llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}')

def pdf_parse_union(
    model_list,
    dataset: Dataset,
    imageWriter,
    parse_mode,
    start_page_id=0,
    end_page_id=None,
    debug_mode=False,
    lang=None,
):
    """解析PDF文档。

    Args:
        model_list: 模型列表
        dataset (Dataset): 数据集对象
        imageWriter: 图像写入器
        parse_mode: 解析模式
        start_page_id (int, optional): 起始页码. Defaults to 0.
        end_page_id (int, optional): 结束页码. Defaults to None.
        debug_mode (bool, optional): 是否开启调试模式. Defaults to False.
        lang (str, optional): 语言代码. Defaults to None.

    Returns:
        dict: 包含解析结果的字典
    """
    pdf_bytes_md5 = compute_md5(dataset.data_bits())
    pdf_info_dict = {}

    # 初始化magic_model
    magic_model = MagicModel(model_list, dataset)

    # 处理页码范围
    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else len(dataset) - 1
    if end_page_id > len(dataset) - 1:
        logger.warning('end_page_id is out of range, use pdf_docs length')
        end_page_id = len(dataset) - 1

    start_time = time.time()

    # 逐页解析
    for page_id, page in enumerate(dataset):
        if debug_mode:
            time_now = time.time()
            logger.info(f'page_id: {page_id}, last_page_cost_time: {round(time.time() - start_time, 2)}')
            start_time = time_now

        if start_page_id <= page_id <= end_page_id:
            page_info = parse_page_core(
                page, magic_model, page_id, pdf_bytes_md5, imageWriter, parse_mode, lang
            )
        else:
            page_info = page.get_page_info()
            page_w = page_info.w
            page_h = page_info.h
            page_info = ocr_construct_page_component_v2(
                [], [], page_id, page_w, page_h, [], [], [], [], [], True, 'skip page'
            )
        pdf_info_dict[f'page_{page_id}'] = page_info

    # 分段处理
    para_split(pdf_info_dict)

    # LLM优化
    process_llm_aided(pdf_info_dict, get_llm_aided_config())

    # 转换格式并清理内存
    pdf_info_list = dict_to_list(pdf_info_dict)
    new_pdf_info_dict = {'pdf_info': pdf_info_list}
    clean_memory(get_device())

    return new_pdf_info_dict


if __name__ == '__main__':
    pass
