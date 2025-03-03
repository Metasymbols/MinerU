from typing import Dict, Optional

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.pre_proc.construct_page_dict import ocr_construct_page_component_v2
from magic_pdf.pre_proc.cut_image import ocr_cut_image_and_table
from magic_pdf.pre_proc.ocr_dict_merge import fill_spans_in_blocks, fix_block_spans_v2, fix_discarded_block
from magic_pdf.pre_proc.ocr_span_list_modify import get_qa_need_list_v2, remove_overlaps_low_confidence_spans, \
    remove_overlaps_min_spans
from magic_pdf.post_proc.para_split_v3 import para_split
from magic_pdf.post_proc.llm_aided import llm_aided_formula, llm_aided_text, llm_aided_title
from magic_pdf.libs.hash_utils import compute_md5
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.libs.config_reader import get_llm_aided_config, get_device
from magic_pdf.libs.convert_utils import dict_to_list
from .base import PDFParser
from .text_processor import TextProcessor


class ContentParser(PDFParser):
    """PDF内容解析器，负责处理文档的内容提取。
    
    此类主要处理PDF文档的内容提取，包括：
    1. 文本内容提取和处理
    2. 图片和表格的处理
    3. 段落分割
    4. LLM优化
    
    继承自PDFParser基类，实现了内容提取相关的具体功能。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_processor = TextProcessor()
        self.pdf_bytes_md5 = compute_md5(self.dataset.data_bits())
    
    def parse_page(self, page_id: int) -> Dict:
        """解析单个页面的内容。

        Args:
            page_id (int): 页面ID

        Returns:
            Dict: 页面内容解析结果
        """
        page = self.dataset[page_id]
        page_info = page.get_page_info()
        page_w = page_info.w
        page_h = page_info.h
        
        # 获取页面布局信息
        layout_info = self.parse_layout(page_id)
        all_bboxes = layout_info['all_bboxes']
        all_discarded_blocks = layout_info['all_discarded_blocks']
        
        # 获取并处理spans
        spans = self.magic_model.get_all_spans(page_id)
        spans = self._process_spans(spans, all_bboxes, all_discarded_blocks, page)
        
        # 处理discarded_blocks
        discarded_block_with_spans, spans = fill_spans_in_blocks(
            all_discarded_blocks, spans, 0.4
        )
        fix_discarded_blocks = fix_discarded_block(discarded_block_with_spans)
        
        # 处理空页面
        if len(all_bboxes) == 0:
            return ocr_construct_page_component_v2(
                [], [], page_id, page_w, page_h, [], [], [],
                [], fix_discarded_blocks, True, 'empty page'
            )
        
        # 处理图像和表格
        spans = ocr_cut_image_and_table(
            spans, page, page_id, self.pdf_bytes_md5, self.image_writer
        )
        
        # 处理blocks
        block_with_spans, spans = fill_spans_in_blocks(all_bboxes, spans, 0.5)
        fix_blocks = fix_block_spans_v2(block_with_spans)
        
        # 获取图片、表格和公式列表
        images, tables, interline_equations = get_qa_need_list_v2(fix_blocks)
        
        # 构造返回结果
        return ocr_construct_page_component_v2(
            fix_blocks, [], page_id, page_w, page_h, [],
            images, tables, interline_equations, fix_discarded_blocks,
            False, []
        )
    
    def parse_document(self, start_page_id: int = 0,
                      end_page_id: Optional[int] = None,
                      debug_mode: bool = False) -> Dict:
        """解析整个文档的内容。

        Args:
            start_page_id (int, optional): 起始页码. Defaults to 0.
            end_page_id (Optional[int], optional): 结束页码. Defaults to None.
            debug_mode (bool, optional): 调试模式. Defaults to False.

        Returns:
            Dict: 文档内容解析结果
        """
        end_page_id = self._validate_page_range(start_page_id, end_page_id)
        pdf_info_dict = {}
        
        for page_id in range(start_page_id, end_page_id + 1):
            if debug_mode:
                self._log_progress(page_id)
            
            page_info = self.parse_page(page_id)
            pdf_info_dict[f'page_{page_id}'] = page_info
        
        # 分段处理
        para_split(pdf_info_dict)
        
        # LLM优化
        self._apply_llm_optimization(pdf_info_dict)
        
        # 转换格式并清理内存
        pdf_info_list = dict_to_list(pdf_info_dict)
        new_pdf_info_dict = {'pdf_info': pdf_info_list}
        clean_memory(get_device())
        
        return new_pdf_info_dict
    
    def _process_spans(self, spans, all_bboxes, all_discarded_blocks, page):
        """处理spans信息。

        Args:
            spans: spans列表
            all_bboxes: 所有边界框
            all_discarded_blocks: 所有丢弃的块
            page: 页面对象

        Returns:
            处理后的spans列表
        """
        # 删除重叠的spans
        spans, _ = remove_overlaps_low_confidence_spans(spans)
        spans, _ = remove_overlaps_min_spans(spans)
        
        # 根据解析模式处理spans
        if self.parse_mode == SupportedPdfParseMethod.TXT:
            spans = self.text_processor.process_text_spans(
                page, spans, all_bboxes, all_discarded_blocks, self.lang
            )
        elif self.parse_mode == SupportedPdfParseMethod.OCR:
            pass
        else:
            raise Exception('parse_mode must be txt or ocr')
        
        return spans
    
    def _apply_llm_optimization(self, pdf_info_dict):
        """应用LLM优化。

        Args:
            pdf_info_dict: PDF信息字典
        """
        llm_config = get_llm_aided_config()
        llm_aided_formula(pdf_info_dict, llm_config)
        llm_aided_text(pdf_info_dict, llm_config)
        llm_aided_title(pdf_info_dict, llm_config)
    
    def _log_progress(self, page_id):
        """记录处理进度。

        Args:
            page_id: 页面ID
        """
        from loguru import logger
        import time
        
        time_now = time.time()
        if hasattr(self, '_start_time'):
            logger.info(
                f'page_id: {page_id}, '
                f'last_page_cost_time: {round(time_now - self._start_time, 2)}'
            )
        self._start_time = time_now