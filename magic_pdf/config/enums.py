
import enum


class SupportedPdfParseMethod(enum.Enum):
    """支持的PDF解析方法枚举类。
    
    定义了系统支持的两种PDF文档解析方式：
    1. OCR方式：通过光学字符识别技术提取PDF中的文本
    2. TXT方式：直接提取PDF中的文本内容
    """
    OCR = 'ocr'
    TXT = 'txt'
