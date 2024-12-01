"""用户输入： model数组，每个元素代表一个页面 pdf在s3的路径 截图保存的s3位置.

然后：
    1）根据s3路径，调用spark集群的api,拿到ak,sk,endpoint，构造出s3PDFReader
    2）根据用户输入的s3地址，调用spark集群的api,拿到ak,sk,endpoint，构造出s3ImageWriter

其余部分至于构造s3cli, 获取ak,sk都在code-clean里写代码完成。不要反向依赖！！！
"""

from loguru import logger
from magic_pdf.data.data_reader_writer import DataWriter
from magic_pdf.libs.version import __version__
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.pdf_parse_by_ocr import parse_pdf_by_ocr
from magic_pdf.pdf_parse_by_txt import parse_pdf_by_txt

PARSE_TYPE_TXT = 'txt'
PARSE_TYPE_OCR = 'ocr'


def parse_txt_pdf(pdf_bytes: bytes, pdf_models: list, imageWriter: DataWriter, is_debug=False,
                  start_page_id=0, end_page_id=None, lang=None,
                  *args, **kwargs):
    """
    解析文本类pdf。

    该函数专门用于解析以文本为主的pdf文件。它通过调用另一个函数parse_pdf_by_txt来提取和处理pdf中的文本信息，
    并根据需要将相关信息添加到解析后的pdf信息字典中。

    参数:
    - pdf_bytes (bytes): pdf文件的字节流。
    - pdf_models (list): 包含pdf解析模型的列表。
    - imageWriter (DataWriter): 用于写入图像数据的对象。
    - is_debug (bool, optional): 是否启用调试模式。默认为False。
    - start_page_id (int, optional): 开始解析的页面ID。默认为0。
    - end_page_id (int, optional): 结束解析的页面ID。默认为None，表示解析到最后一页。
    - lang (str, optional): pdf文档的语言。默认为None。

    返回:
    - dict: 包含pdf解析后信息的字典。
    """
    # 调用parse_pdf_by_txt函数进行pdf解析
    pdf_info_dict = parse_pdf_by_txt(
        pdf_bytes,
        pdf_models,
        imageWriter,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        debug_mode=is_debug,
        lang=lang,
    )

    # 添加解析类型到pdf信息字典
    pdf_info_dict['_parse_type'] = PARSE_TYPE_TXT

    # 添加版本号到pdf信息字典
    pdf_info_dict['_version_name'] = __version__

    # 如果指定了语言，添加语言信息到pdf信息字典
    if lang is not None:
        pdf_info_dict['_lang'] = lang

    # 返回包含解析信息的pdf信息字典
    return pdf_info_dict


def parse_ocr_pdf(pdf_bytes: bytes, pdf_models: list, imageWriter: DataWriter, is_debug=False,
                  start_page_id=0, end_page_id=None, lang=None,
                  *args, **kwargs):
    """
    解析ocr类pdf。

    该函数通过光学字符识别（OCR）技术解析PDF文档的页面，将识别到的信息存储在提供的数据结构中。
    它允许配置起始和结束页面ID以处理大型文档的部分内容，选择是否启用调试模式，以及指定识别使用的语言。

    参数:
    - pdf_bytes (bytes): PDF文档的字节流。
    - pdf_models (list): 包含OCR模型的列表，用于PDF解析。
    - imageWriter (DataWriter): 一个DataWriter实例，用于写入处理过程中的图像数据。
    - is_debug (bool, optional): 调试模式开关，默认为False。在调试模式下，可能会输出额外的日志信息。
    - start_page_id (int, optional): 开始解析的页面ID，默认为0，表示从第一个页面开始。
    - end_page_id (int, optional): 结束解析的页面ID，默认为None，表示解析到最后一个页面。
    - lang (str, optional): 指定OCR解析使用的语言代码，如'en'、'zh-cn'等，默认为None，表示使用默认语言设置。
    - *args, **kwargs: 允许函数接受额外的参数和关键字参数，提供灵活性。

    返回:
    - pdf_info_dict (dict): 包含OCR解析信息的字典，包括解析类型、版本号等元数据。
    """

    # 通过OCR技术解析PDF，获取包含解析信息的字典
    pdf_info_dict = parse_pdf_by_ocr(
        pdf_bytes,
        pdf_models,
        imageWriter,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        debug_mode=is_debug,
        lang=lang,
    )

    # 添加解析类型到返回的字典中，标识为OCR解析
    pdf_info_dict['_parse_type'] = PARSE_TYPE_OCR

    # 添加版本号到返回的字典中，记录解析时使用的版本
    pdf_info_dict['_version_name'] = __version__

    # 如果指定了语言，将其添加到返回的字典中
    if lang is not None:
        pdf_info_dict['_lang'] = lang

    # 返回包含所有解析信息和元数据的字典
    return pdf_info_dict


def parse_union_pdf(pdf_bytes: bytes, pdf_models: list, imageWriter: DataWriter, is_debug=False,
                    input_model_is_empty: bool = False,
                    start_page_id=0, end_page_id=None, lang=None,
                    *args, **kwargs):
    """ocr和文本混合的pdf，全部解析出来."""

    def parse_pdf(method):
        try:
            return method(
                pdf_bytes,
                pdf_models,
                imageWriter,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                debug_mode=is_debug,
                lang=lang,
            )
        except Exception as e:
            logger.exception(e)
            return None

    pdf_info_dict = parse_pdf(parse_pdf_by_txt)
    if pdf_info_dict is None or pdf_info_dict.get('_need_drop', False):
        logger.warning(
            'parse_pdf_by_txt drop or error, switch to parse_pdf_by_ocr')
        if input_model_is_empty:
            layout_model = kwargs.get('layout_model', None)
            formula_enable = kwargs.get('formula_enable', None)
            table_enable = kwargs.get('table_enable', None)
            pdf_models = doc_analyze(
                pdf_bytes,
                ocr=True,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                lang=lang,
                layout_model=layout_model,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )
        pdf_info_dict = parse_pdf(parse_pdf_by_ocr)
        if pdf_info_dict is None:
            raise Exception(
                'Both parse_pdf_by_txt and parse_pdf_by_ocr failed.')
        else:
            pdf_info_dict['_parse_type'] = PARSE_TYPE_OCR
    else:
        pdf_info_dict['_parse_type'] = PARSE_TYPE_TXT

    pdf_info_dict['_version_name'] = __version__

    if lang is not None:
        pdf_info_dict['_lang'] = lang

    return pdf_info_dict
