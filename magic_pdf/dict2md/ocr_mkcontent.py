import re

from loguru import logger

from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.libs.commons import join_path
from magic_pdf.libs.language import detect_lang
from magic_pdf.libs.markdown_utils import ocr_escape_special_markdown_char
from magic_pdf.post_proc.para_split_v3 import ListLineTag


def __is_hyphen_at_line_end(line):
    """检查一行文本是否以字母加连字符结尾。

    用于判断英文单词是否因换行而被分割，需要在处理时特别处理这种情况。

    参数:
        line (str): 需要检查的文本行

    返回:
        bool: 如果文本行以一个或多个字母后跟连字符结尾则返回True，否则返回False

    示例:
        >>> __is_hyphen_at_line_end("word-")
        True
        >>> __is_hyphen_at_line_end("word")
        False
    """
    # 使用正则表达式检查文本行是否以字母加连字符结尾
    return bool(re.search(r'[A-Za-z]+-\s*$', line))


def ocr_mk_mm_markdown_with_para_and_pagination(pdf_info_dict: list, img_buket_path: str) -> list:
    """将PDF信息转换为带分页和段落的Markdown内容。

    将OCR识别后的PDF信息字典转换为带有页码和段落的Markdown格式内容。每个页面的内容会被单独处理，
    并保持原有的段落结构。支持图片、表格、公式等多媒体内容的转换。

    参数:
        pdf_info_dict (list): 包含PDF页面信息的字典列表，每个字典包含一个页面的完整信息
        img_buket_path (str): 图片引用的基础路径，用于构建图片的完整URL

    返回:
        list: 包含页码和对应Markdown内容的字典列表，每个字典包含以下字段：
            - page_no: 页码（从0开始）
            - md_content: 该页面的Markdown格式内容

    示例:
        >>> result = ocr_mk_mm_markdown_with_para_and_pagination(pdf_dict, "/path/to/images")
        >>> print(result[0])
        {"page_no": 0, "md_content": "# 标题\n\n正文内容..."}
    """
    markdown_with_para_and_pagination = []
    for page_no, page_info in enumerate(pdf_info_dict):
        paras_of_layout = page_info.get('para_blocks')
        if not paras_of_layout:
            markdown_with_para_and_pagination.append({
                'page_no': page_no,
                'md_content': ''
            })
            continue

        page_markdown = ocr_mk_markdown_with_para_core_v2(
            paras_of_layout, 'mm', img_buket_path)
        markdown_with_para_and_pagination.append({
            'page_no': page_no,
            'md_content': '\n\n'.join(page_markdown)
        })

    return markdown_with_para_and_pagination


def ocr_mk_markdown_with_para_core_v2(paras_of_layout,
                                      mode,
                                      img_buket_path='',
                                      ):
    page_markdown = []
    for para_block in paras_of_layout:
        para_text = ''
        para_type = para_block['type']
        if para_type in [BlockType.Text, BlockType.List, BlockType.Index]:
            para_text = merge_para_with_text(para_block)
        elif para_type == BlockType.Title:
            title_level = get_title_level(para_block)
            para_text = f'{"#" * title_level} {merge_para_with_text(para_block)}'
        elif para_type == BlockType.InterlineEquation:
            para_text = merge_para_with_text(para_block)
        elif para_type == BlockType.Image:
            if mode == 'nlp':
                continue
            elif mode == 'mm':
                for block in para_block['blocks']:  # 1st.拼image_body
                    if block['type'] == BlockType.ImageBody:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.Image:
                                    if span.get('image_path', ''):
                                        para_text += f"\n![]({join_path(img_buket_path, span['image_path'])})  \n"
                for block in para_block['blocks']:  # 2nd.拼image_caption
                    if block['type'] == BlockType.ImageCaption:
                        para_text += merge_para_with_text(block) + '  \n'
                for block in para_block['blocks']:  # 3rd.拼image_footnote
                    if block['type'] == BlockType.ImageFootnote:
                        para_text += merge_para_with_text(block) + '  \n'
        elif para_type == BlockType.Table:
            if mode == 'nlp':
                continue
            elif mode == 'mm':
                for block in para_block['blocks']:  # 1st.拼table_caption
                    if block['type'] == BlockType.TableCaption:
                        para_text += merge_para_with_text(block) + '  \n'
                for block in para_block['blocks']:  # 2nd.拼table_body
                    if block['type'] == BlockType.TableBody:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.Table:
                                    # if processed by table model
                                    if span.get('latex', ''):
                                        para_text += f"\n\n$\n {span['latex']}\n$\n\n"
                                    elif span.get('html', ''):
                                        para_text += f"\n\n{span['html']}\n\n"
                                    elif span.get('image_path', ''):
                                        para_text += f"\n![]({join_path(img_buket_path, span['image_path'])})  \n"
                for block in para_block['blocks']:  # 3rd.拼table_footnote
                    if block['type'] == BlockType.TableFootnote:
                        para_text += merge_para_with_text(block) + '  \n'

        if para_text.strip() == '':
            continue
        else:
            page_markdown.append(para_text.strip() + '  ')

    return page_markdown


def detect_language(text):
    """检测文本的主要语言类型。

    通过分析文本中英文字符的比例来判断文本的主要语言。如果英文字符占比超过50%，则认为是英文文本，
    否则认为是其他语言。这个简单的判断方法主要用于确定文本的换行处理策略。

    参数:
        text (str): 需要检测语言类型的文本

    返回:
        str: 返回检测到的语言类型：
            - 'en': 英文（英文字符占比>=50%）
            - 'unknown': 未知语言（英文字符占比<50%）
            - 'empty': 空文本

    示例:
        >>> detect_language("Hello World")
        'en'
        >>> detect_language("你好世界")
        'unknown'
        >>> detect_language("")
        'empty'
    """
    en_pattern = r'[a-zA-Z]+'
    en_matches = re.findall(en_pattern, text)
    en_length = sum(len(match) for match in en_matches)
    if len(text) > 0:
        if en_length / len(text) >= 0.5:
            return 'en'
        else:
            return 'unknown'
    else:
        return 'empty'


def full_to_half(text: str) -> str:
    """将全角字符转换为半角字符。

    将文本中的全角字符（如全角数字、字母、标点符号等）转换为对应的半角字符。
    转换是通过Unicode码点操作实现的，主要处理以下情况：
    1. 全角ASCII字符（FF01-FF5E）转换为对应的半角字符
    2. 全角空格（0x3000）转换为半角空格

    参数:
        text (str): 包含全角字符的字符串

    返回:
        str: 转换后的字符串，其中全角字符已被转换为对应的半角字符

    示例:
        >>> full_to_half("Ｈｅｌｌｏ　Ｗｏｒｌｄ！")
        'Hello World!'
    """
    result = []
    for char in text:
        code = ord(char)
        # Full-width ASCII variants (FF01-FF5E)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        # Full-width space
        elif code == 0x3000:
            result.append(' ')
        else:
            result.append(char)
    return ''.join(result)


def __process_text_span(span, block_lang, is_last_span=False):
    """处理文本片段并根据语言上下文进行格式化。

    根据文本片段的类型（普通文本、行内公式、行间公式）和语言环境（中日韩、英文等）
    对文本进行相应的处理和格式化。主要处理以下情况：
    1. 普通文本：转义Markdown特殊字符
    2. 行内公式：添加$符号
    3. 行间公式：添加$$符号和换行
    4. 根据语言类型处理空格和换行

    参数:
        span (dict): 包含内容和类型信息的文本片段字典
        block_lang (str): 检测到的文本块语言类型
        is_last_span (bool): 是否是行内最后一个片段

    返回:
        str: 处理和格式化后的文本内容

    示例:
        >>> span = {"type": "text", "content": "Hello"}
        >>> __process_text_span(span, "en", False)
        'Hello '
    """
    content = ''
    span_type = span['type']
    
    if span_type == ContentType.Text:
        content = ocr_escape_special_markdown_char(span['content'])
    elif span_type == ContentType.InlineEquation:
        content = f"${span['content']}$"
    elif span_type == ContentType.InterlineEquation:
        content = f"\n$$\n{span['content']}\n$$\n"

    content = content.strip()
    if not content:
        return ''

    langs = ['zh', 'ja', 'ko']
    if block_lang in langs:
        # 中文/日语/韩文语境下，换行不需要空格分隔
        if is_last_span and span_type not in [ContentType.InlineEquation]:
            return content
        return f'{content} '
    else:
        if span_type in [ContentType.Text, ContentType.InlineEquation]:
            # 处理英文连字符
            if is_last_span and span_type == ContentType.Text and __is_hyphen_at_line_end(content):
                return content[:-1]
            return f'{content} '
        elif span_type == ContentType.InterlineEquation:
            return content
    return ''

def merge_para_with_text(para_block):
    """将段落块合并为格式化文本，处理不同的语言和内容类型。

    将包含多个文本行和片段的段落块合并成一个格式化的文本字符串。处理过程包括：
    1. 检测段落的主要语言
    2. 将全角字符转换为半角字符
    3. 处理列表项的特殊格式
    4. 根据语言环境处理文本片段之间的空格和换行

    参数:
        para_block (dict): 包含行和文本片段的段落块字典

    返回:
        str: 合并和格式化后的段落文本

    示例:
        >>> para = {"lines": [{"spans": [{"type": "text", "content": "Hello"}]}]}
        >>> merge_para_with_text(para)
        'Hello'
    """
    # 获取段落的主要语言
    block_text = ''
    for line in para_block['lines']:
        for span in line['spans']:
            if span['type'] in [ContentType.Text]:
                span['content'] = full_to_half(span['content'])
                block_text += span['content']
    block_lang = detect_lang(block_text)

    # 处理段落文本
    para_text = ''
    for i, line in enumerate(para_block['lines']):
        # 处理列表项
        if i >= 1 and line.get(ListLineTag.IS_LIST_START_LINE, False):
            para_text += '  \n'

        # 处理每一行的span
        for j, span in enumerate(line['spans']):
            is_last_span = (j == len(line['spans']) - 1)
            para_text += __process_text_span(span, block_lang, is_last_span)

    return para_text

def para_to_standard_format_v2(para_block, img_buket_path, page_idx, drop_reason=None):
    """将段落块转换为标准格式。

    将不同类型的段落块（文本、标题、公式、图片、表格等）转换为统一的标准格式。
    支持以下类型的转换：
    1. 文本/列表/索引 -> 标准文本格式
    2. 标题 -> 带层级的文本格式
    3. 行间公式 -> LaTeX格式
    4. 图片 -> 带路径和说明的图片格式
    5. 表格 -> 带标题和脚注的表格格式

    参数:
        para_block (dict): 需要转换的段落块
        img_buket_path (str): 图片存储的基础路径
        page_idx (int): 页面索引
        drop_reason (str, optional): 丢弃原因，默认为None

    返回:
        dict: 标准格式的内容字典，包含类型、内容和其他相关信息

    示例:
        >>> block = {"type": "text", "lines": [{"spans": [{"type": "text", "content": "Hello"}]}]}
        >>> para_to_standard_format_v2(block, "/images", 0)
        {"type": "text", "text": "Hello", "page_idx": 0}
    """
    para_type = para_block['type']
    para_content = {}
    if para_type in [BlockType.Text, BlockType.List, BlockType.Index]:
        para_content = {
            'type': 'text',
            'text': merge_para_with_text(para_block),
        }
    elif para_type == BlockType.Title:
        title_level = get_title_level(para_block)
        para_content = {
            'type': 'text',
            'text': merge_para_with_text(para_block),
            'text_level': title_level,
        }
    elif para_type == BlockType.InterlineEquation:
        para_content = {
            'type': 'equation',
            'text': merge_para_with_text(para_block),
            'text_format': 'latex',
        }
    elif para_type == BlockType.Image:
        para_content = {'type': 'image', 'img_path': '', 'img_caption': [], 'img_footnote': []}
        for block in para_block['blocks']:
            if block['type'] == BlockType.ImageBody:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.Image:
                            if span.get('image_path', ''):
                                para_content['img_path'] = join_path(img_buket_path, span['image_path'])
            if block['type'] == BlockType.ImageCaption:
                para_content['img_caption'].append(merge_para_with_text(block))
            if block['type'] == BlockType.ImageFootnote:
                para_content['img_footnote'].append(merge_para_with_text(block))
    elif para_type == BlockType.Table:
        para_content = {'type': 'table', 'img_path': '', 'table_caption': [], 'table_footnote': []}
        for block in para_block['blocks']:
            if block['type'] == BlockType.TableBody:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.Table:

                            if span.get('latex', ''):
                                para_content['table_body'] = f"\n\n$\n {span['latex']}\n$\n\n"
                            elif span.get('html', ''):
                                para_content['table_body'] = f"\n\n{span['html']}\n\n"

                            if span.get('image_path', ''):
                                para_content['img_path'] = join_path(img_buket_path, span['image_path'])

            if block['type'] == BlockType.TableCaption:
                para_content['table_caption'].append(merge_para_with_text(block))
            if block['type'] == BlockType.TableFootnote:
                para_content['table_footnote'].append(merge_para_with_text(block))

    para_content['page_idx'] = page_idx

    if drop_reason is not None:
        para_content['drop_reason'] = drop_reason

    return para_content


def union_make(pdf_info_dict: list,
               make_mode: str,
               drop_mode: str,
               img_buket_path: str = '',
               ):
    """统一处理PDF内容，支持多种输出模式和丢弃策略。

    根据指定的模式和策略处理PDF信息字典，支持以下功能：
    1. 多种输出模式：
       - MM_MD: 多媒体Markdown格式，包含图片和表格
       - NLP_MD: 纯文本Markdown格式，适合NLP处理
       - STANDARD_FORMAT: 标准JSON格式
    2. 多种丢弃策略：
       - NONE: 不丢弃任何内容
       - NONE_WITH_REASON: 保留内容但标记丢弃原因
       - WHOLE_PDF: 整个PDF不可用时抛出异常
       - SINGLE_PAGE: 跳过不可用的页面

    参数:
        pdf_info_dict (list): PDF信息字典列表
        make_mode (str): 输出模式，可选值：MM_MD、NLP_MD、STANDARD_FORMAT
        drop_mode (str): 丢弃策略，可选值：NONE、NONE_WITH_REASON、WHOLE_PDF、SINGLE_PAGE
        img_buket_path (str, optional): 图片存储的基础路径，默认为空字符串

    返回:
        Union[str, list]: 根据make_mode返回不同格式的内容：
            - MM_MD/NLP_MD: 返回合并后的Markdown字符串
            - STANDARD_FORMAT: 返回标准格式的内容列表

    示例:
        >>> result = union_make(pdf_dict, "MM_MD", "NONE", "/images")
        >>> print(type(result))
        <class 'str'>
    """
    output_content = []
    for page_info in pdf_info_dict:
        drop_reason_flag = False
        drop_reason = None
        if page_info.get('need_drop', False):
            drop_reason = page_info.get('drop_reason')
            if drop_mode == DropMode.NONE:
                pass
            elif drop_mode == DropMode.NONE_WITH_REASON:
                drop_reason_flag = True
            elif drop_mode == DropMode.WHOLE_PDF:
                raise Exception((f'drop_mode is {DropMode.WHOLE_PDF} ,'
                                 f'drop_reason is {drop_reason}'))
            elif drop_mode == DropMode.SINGLE_PAGE:
                logger.warning((f'drop_mode is {DropMode.SINGLE_PAGE} ,'
                                f'drop_reason is {drop_reason}'))
                continue
            else:
                raise Exception('drop_mode can not be null')

        paras_of_layout = page_info.get('para_blocks')
        page_idx = page_info.get('page_idx')
        if not paras_of_layout:
            continue
        if make_mode == MakeMode.MM_MD:
            page_markdown = ocr_mk_markdown_with_para_core_v2(
                paras_of_layout, 'mm', img_buket_path)
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.NLP_MD:
            page_markdown = ocr_mk_markdown_with_para_core_v2(
                paras_of_layout, 'nlp')
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.STANDARD_FORMAT:
            for para_block in paras_of_layout:
                if drop_reason_flag:
                    para_content = para_to_standard_format_v2(
                        para_block, img_buket_path, page_idx)
                else:
                    para_content = para_to_standard_format_v2(
                        para_block, img_buket_path, page_idx)
                output_content.append(para_content)
    if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
        return '\n\n'.join(output_content)
    elif make_mode == MakeMode.STANDARD_FORMAT:
        return output_content


def get_title_level(block: dict) -> int:
    """获取标题的层级，确保在有效范围内（1-4）。

    Args:
        block (dict): Block containing title information

    Returns:
        int: Title level between 1 and 4
    """
    title_level = block.get('level', 1)
    return max(1, min(4, title_level))