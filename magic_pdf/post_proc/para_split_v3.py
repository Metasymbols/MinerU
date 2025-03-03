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
        分组后的blocks列表，每个分组包含相邻的文本块，由标题或行间公式分隔
    """
    result = []
    current_group = []

def __is_list_or_index_block(block: Dict[str, Any]) -> str:
    """判断block是否为列表或索引类型
    
    Args:
        block: 需要判断的block，包含行信息和边界框信息
        
    Returns:
        block的类型，可能为'text'、'list'或'index'
    """
    if len(block['lines']) < 2:
        return BlockType.Text

def para_split(pdf_info_dict: Dict[int, Dict[str, Any]]) -> None:
    """分割PDF文档的段落，将相邻的文本块合并成段落
    
    Args:
        pdf_info_dict: PDF文档信息字典，包含每页的预处理块信息
        
    Note:
        函数会修改输入的pdf_info_dict，在其中添加para_blocks字段
        para_blocks包含合并后的段落信息，每个段落包含页码、页面大小等信息
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
