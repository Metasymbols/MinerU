import copy

from loguru import logger

from magic_pdf.config.constants import CROSS_PAGE, LINES_DELETED
from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.libs.language import detect_lang

LINE_STOP_FLAG = (
    '.',
    '!',
    '?',
    '。',
    '！',
    '？',
    ')',
    '）',
    '"',
    '”',
    ':',
    '：',
    ';',
    '；',
)
LIST_END_FLAG = ('.', '。', ';', '；')

LINE_HEIGHT_RATIO = 0.5
LINE_OVERLAP_THRESHOLD = 0.7
BLOCK_WEIGHT_THRESHOLD_WIDE = 0.26
BLOCK_WEIGHT_THRESHOLD_NARROW = 0.36
BLOCK_WEIGHT_RATIO_THRESHOLD = 0.5
LIST_RATIO_THRESHOLD = 0.8
BLOCK_HEIGHT_RATIO_THRESHOLD = 0.4
RIGHT_MARGIN_RATIO = 0.1

class ListLineTag:
    IS_LIST_START_LINE = 'is_list_start_line'
    IS_LIST_END_LINE = 'is_list_end_line'

def __process_blocks(blocks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """对所有block预处理
    1. 通过title和interline_equation将block分组
    2. bbox边界根据line信息重置
    
    Args:
        blocks: 需要处理的blocks列表
        
    Returns:
        分组后的blocks列表
    """
    result = []
    current_group = []

    for i, current_block in enumerate(blocks):
        if current_block['type'] == 'text':
            current_block['bbox_fs'] = copy.deepcopy(current_block['bbox'])
            if 'lines' in current_block and current_block['lines']:
                lines = current_block['lines']
                current_block['bbox_fs'] = [
                    min(line['bbox'][0] for line in lines),
                    min(line['bbox'][1] for line in lines),
                    max(line['bbox'][2] for line in lines),
                    max(line['bbox'][3] for line in lines),
                ]
            current_group.append(current_block)

        if i + 1 < len(blocks):
            next_block = blocks[i + 1]
            if next_block['type'] in ['title', 'interline_equation']:
                result.append(current_group)
                current_group = []

    if current_group:
        result.append(current_group)

    return result

def __is_line_aligned_left(line_bbox: List[float], block_bbox: List[float], line_height: float) -> bool:
    """判断行是否左对齐"""
    return abs(block_bbox[0] - line_bbox[0]) < line_height * LINE_HEIGHT_RATIO

def __is_line_aligned_right(line_bbox: List[float], block_bbox: List[float], line_height: float) -> bool:
    """判断行是否右对齐"""
    return abs(block_bbox[2] - line_bbox[2]) < line_height

def __get_line_text(line: Dict[str, Any]) -> str:
    """获取行文本内容"""
    return ''.join(span['content'].strip() for span in line['spans'] 
                   if span['type'] == ContentType.Text)

def __is_list_or_index_block(block: Dict[str, Any]) -> str:
    """判断block是否为列表或索引类型
    
    Args:
        block: 需要判断的block
        
    Returns:
        block的类型
    """
    if len(block['lines']) < 2:
        return BlockType.Text
        
    first_line = block['lines'][0]
    last_line = block['lines'][-1]
    line_height = first_line['bbox'][3] - first_line['bbox'][1]
    block_bbox = block['bbox_fs']
    block_width = block_bbox[2] - block_bbox[0]
    block_height = block_bbox[3] - block_bbox[1]
    page_width, _ = block['page_size']
    
    block_width_ratio = page_width and block_width / page_width or 0
    
    # 统计行对齐信息
    alignment_stats = {
        'left_aligned': 0,
        'left_indented': 0,
        'right_aligned': 0,
        'right_indented': 0,
        'center_aligned': 0,
        'both_sides_indented': 0
    }
    
    lines_text = []
    for line in block['lines']:
        line_bbox = line['bbox']
        line_mid_x = (line_bbox[0] + line_bbox[2]) / 2
        block_mid_x = (block_bbox[0] + block_bbox[2]) / 2

def __para_merge_page(blocks: List[Dict[str, Any]]) -> None:
    page_text_blocks_groups = __process_blocks(blocks)
    
    for text_blocks_group in page_text_blocks_groups:
        if not text_blocks_group:
            continue
            
        # 在合并前对所有block判断类型
        for block in text_blocks_group:
            block['type'] = __is_list_or_index_block(block)

        if len(text_blocks_group) <= 1:
            continue
            
        # 判断是否为列表组
        is_list_group = __is_list_group(text_blocks_group)

        # 倒序遍历处理相邻块
        for i in range(len(text_blocks_group) - 1, 0, -1):
            current_block = text_blocks_group[i]
            prev_block = text_blocks_group[i - 1]

            if (current_block['type'] == 'text' and 
                prev_block['type'] == 'text' and 
                not is_list_group):
                __merge_2_text_blocks(current_block, prev_block)
            elif __can_merge_list_blocks(current_block, prev_block):
                __merge_2_list_blocks(current_block, prev_block)

def __can_merge_list_blocks(block1: Dict[str, Any], block2: Dict[str, Any]) -> bool:
    """判断两个列表块是否可以合并"""
    return ((block1['type'] == BlockType.List and block2['type'] == BlockType.List) or
            (block1['type'] == BlockType.Index and block2['type'] == BlockType.Index))

def para_split(pdf_info_dict: Dict[int, Dict[str, Any]]) -> None:
    """分割PDF文档的段落
    
    Args:
        pdf_info_dict: PDF文档信息字典
    """
    # 收集所有块
    all_blocks = []
    for page_num, page in pdf_info_dict.items():
        blocks = copy.deepcopy(page['preproc_blocks'])
        for block in blocks:
            block['page_num'] = page_num
            block['page_size'] = page['page_size']
        all_blocks.extend(blocks)

    # 合并页面段落
    __para_merge_page(all_blocks)
    
    # 更新页面块信息
    for page_num, page in pdf_info_dict.items():
        page['para_blocks'] = [block for block in all_blocks if block['page_num'] == page_num]

if __name__ == '__main__':
    input_blocks: List[Dict[str, Any]] = []
    groups = __process_blocks(input_blocks)
    for group_index, group in enumerate(groups):
        print(f'Group {group_index}: {group}')
