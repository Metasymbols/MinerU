from typing import Dict, List, Optional
from .base import PDFParser
from magic_pdf.config.ocr_content_type import BlockType
from magic_pdf.libs.language import detect_language


class TextProcessor(PDFParser):
    """文本处理器，负责处理PDF文档中的文本内容。
    
    此类主要负责：
    1. 文本内容的提取和处理
    2. 文本语言检测
    3. 文本格式优化
    4. 特殊字符处理
    
    继承自PDFParser基类，实现了文本处理相关的具体功能。
    """
    
    def parse_page(self, page_id: int) -> Dict:
        """解析单个页面的文本内容。

        Args:
            page_id (int): 页面ID

        Returns:
            Dict: 页面文本解析结果
        """
        page = self.dataset[page_id]
        page_info = page.get_page_info()
        
        # 获取页面文本内容
        text_blocks = self.magic_model.get_text_blocks(page_id)
        processed_blocks = self._process_text_blocks(text_blocks)
        
        return {
            'page_id': page_id,
            'page_w': page_info.w,
            'page_h': page_info.h,
            'text_blocks': processed_blocks
        }
    
    def parse_document(self, start_page_id: int = 0,
                      end_page_id: Optional[int] = None,
                      debug_mode: bool = False) -> Dict:
        """解析整个文档的文本内容。

        Args:
            start_page_id (int, optional): 起始页码. Defaults to 0.
            end_page_id (Optional[int], optional): 结束页码. Defaults to None.
            debug_mode (bool, optional): 调试模式. Defaults to False.

        Returns:
            Dict: 文档文本解析结果
        """
        end_page_id = self._validate_page_range(start_page_id, end_page_id)
        text_results = {}
        
        for page_id in range(start_page_id, end_page_id + 1):
            text_results[f'page_{page_id}'] = self.parse_page(page_id)
        
        return text_results
    
    def process_text_spans(self, page, spans: List, all_bboxes: List,
                         all_discarded_blocks: List, lang: Optional[str] = None) -> List:
        """处理文本spans。

        Args:
            page: 页面对象
            spans (List): spans列表
            all_bboxes (List): 所有边界框
            all_discarded_blocks (List): 所有丢弃的块
            lang (Optional[str], optional): 语言代码. Defaults to None.

        Returns:
            List: 处理后的spans列表
        """
        processed_spans = []
        
        for span in spans:
            # 检测语言
            if not lang:
                lang = detect_language(span.text)
            
            # 处理文本
            processed_text = self._process_text(span.text, lang)
            span.text = processed_text
            processed_spans.append(span)
        
        return processed_spans
    
    def _process_text_blocks(self, blocks: List) -> List:
        """处理文本块。

        Args:
            blocks (List): 文本块列表

        Returns:
            List: 处理后的文本块列表
        """
        processed_blocks = []
        
        for block in blocks:
            if block.type == BlockType.TEXT:
                # 处理文本内容
                processed_text = self._process_text(block.text, self.lang)
                block.text = processed_text
            processed_blocks.append(block)
        
        return processed_blocks
    
    def _process_text(self, text: str, lang: Optional[str] = None) -> str:
        """处理文本内容。

        Args:
            text (str): 原始文本
            lang (Optional[str], optional): 语言代码. Defaults to None.

        Returns:
            str: 处理后的文本
        """
        if not text:
            return text
        
        # 移除多余空白字符
        text = ' '.join(text.split())
        
        # 根据语言进行特殊处理
        if lang == 'zh':
            text = self._process_chinese_text(text)
        elif lang == 'en':
            text = self._process_english_text(text)
        
        return text
    
    def _process_chinese_text(self, text: str) -> str:
        """处理中文文本。

        Args:
            text (str): 中文文本

        Returns:
            str: 处理后的中文文本
        """
        # 移除中文标点前后的空格
        text = text.replace(' 。', '。').replace('。 ', '。')
        text = text.replace(' ，', '，').replace('，', '，')
        text = text.replace(' ：', '：').replace('： ', '：')
        text = text.replace(' ；', '；').replace('； ', '；')
        
        return text
    
    def _process_english_text(self, text: str) -> str:
        """处理英文文本。

        Args:
            text (str): 英文文本

        Returns:
            str: 处理后的英文文本
        """
        # 确保标点符号后有空格
        text = text.replace('.','. ').replace(',',', ')
        text = text.replace('!','! ').replace('?','? ')
        text = text.replace(';','; ').replace(':',': ')
        
        # 移除多余空格
        text = ' '.join(text.split())
        
        return text