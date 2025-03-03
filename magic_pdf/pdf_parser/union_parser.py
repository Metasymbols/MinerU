from typing import Dict, List, Optional
from loguru import logger
import time

from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.config.ocr_content_type import BlockType
from magic_pdf.data.dataset import Dataset
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.libs.config_reader import get_llm_aided_config, get_device
from magic_pdf.libs.convert_utils import dict_to_list
from magic_pdf.libs.hash_utils import compute_md5
from magic_pdf.model.magic_model import MagicModel
from magic_pdf.post_proc.llm_aided import llm_aided_formula, llm_aided_text, llm_aided_title
from magic_pdf.post_proc.para_split_v3 import para_split
from magic_pdf.pre_proc.construct_page_dict import ocr_construct_page_component_v2
from magic_pdf.pre_proc.cut_image import ocr_cut_image_and_table
from magic_pdf.pre_proc.ocr_detect_all_bboxes import ocr_prepare_bboxes_for_layout_split_v2
from magic_pdf.pre_proc.ocr_dict_merge import fill_spans_in_blocks, fix_block_spans_v2, fix_discarded_block
from magic_pdf.pre_proc.ocr_span_list_modify import get_qa_need_list_v2, remove_overlaps_low_confidence_spans, \
    remove_overlaps_min_spans

from .base import PDFParser


class UnionParser(PDFParser):
    """统一PDF解析器，整合了布局分析和内容提取功能。
    
    此类继承自PDFParser基类，实现了完整的PDF解析功能，包括：
    1. 页面布局分析
    2. 文本内容提取
    3. 图片和表格处理
    4. 段落分割
    5. LLM优化
    
    Attributes:
        pdf_bytes_md5 (str): PDF文件的MD5值
    """
    
    def __init__(self, dataset: Dataset, magic_model: MagicModel, image_writer,
                 parse_mode: str, lang: Optional[str] = None):
        """初始化统一PDF解析器。

        Args:
            dataset (Dataset): PDF数据集对象
            magic_model (MagicModel): 魔法模型对象
            image_writer: 图像写入器
            parse_mode (str): 解析模式
            lang (Optional[str], optional): 语言代码. Defaults to None.
        """
        super().__init__(dataset, magic_model, image_writer, parse_mode, lang)
        self.pdf_bytes_md5 = compute_md5(dataset.data_bits())
    
    def parse_page(self, page_id: int) -> Dict:
        """解析单个页面。

        Args:
            page_id (int): 页面ID

        Returns:
            Dict: 页面解析结果
        """
        page = self.dataset[page_id]
        page_info = page.get_page_info()
        page_w = page_info.w
        page_h = page_info.h
        
        # 获取页面基本信息和区块
        img_groups = self.magic_model.get_imgs_v2(page_id)
        table_groups = self.magic_model.get_tables_v2(page_id)
        discarded_blocks = self.magic_model.get_discarded(page_id)
        text_blocks = self.magic_model.get_text_blocks(page_id)
        title_blocks = self.magic_model.get_title_blocks(page_id)
        _, interline_equations, _ = self.magic_model.get_equations(page_id)
        
        # 处理图片和表格区块
        img_body_blocks, img_caption_blocks, img_footnote_blocks = self._process_groups(
            img_groups, 'image_body', 'image_caption_list', 'image_footnote_list'
        )
        
        table_body_blocks, table_caption_blocks, table_footnote_blocks = self._process_groups(
            table_groups, 'table_body', 'table_caption_list', 'table_footnote_list'
        )
        
        # 整理区块边界框
        all_bboxes, all_discarded_blocks = ocr_prepare_bboxes_for_layout_split_v2(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks, text_blocks, title_blocks, interline_equations,
            page_w, page_h
        )
        
        # 处理spans信息
        spans = self.magic_model.get_all_spans(page_id)
        spans = self._process_spans(spans, all_bboxes, all_discarded_blocks)
        
        # 处理discarded_blocks
        discarded_block_with_spans, spans = fill_spans_in_blocks(
            all_discarded_blocks, spans, 0.4
        )
        fix_discarded_blocks = fix_discarded_block(discarded_block_with_spans)
        
        # 处理空页面
        if len(all_bboxes) == 0:
            logger.warning(f'skip this page, not found useful bbox, page_id: {page_id}')
            return ocr_construct_page_component_v2(
                [], [], page_id, page_w, page_h, [], [], [],
                interline_equations, fix_discarded_blocks, True, 'empty page'
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
        """解析整个文档。

        Args:
            start_page_id (int, optional): 起始页码. Defaults to 0.
            end_page_id (Optional[int], optional): 结束页码. Defaults to None.
            debug_mode (bool, optional): 调试模式. Defaults to False.

        Returns:
            Dict: 文档解析结果
        """
        end_page_id = self._validate_page_range(start_page_id, end_page_id)
        pdf_info_dict = {}
        start_time = time.time()
        
        for page_id in range(start_page_id, end_page_id + 1):
            if debug_mode:
                time_now = time.time()
                logger.info(f'page_id: {page_id}, last_page_cost_time: {round(time.time() - start_time, 2)}')
                start_time = time_now
            
            page_info = self.parse_page(page_id)
            pdf_info_dict[f'page_{page_id}'] = page_info
        
        # 分段处理
        para_split(pdf_info_dict)
        
        # LLM优化
        llm_config = get_llm_aided_config()
        llm_aided_formula(pdf_info_dict, llm_config)
        llm_aided_text(pdf_info_dict, llm_config)
        llm_aided_title(pdf_info_dict, llm_config)
        
        # 转换格式并清理内存
        pdf_info_list = dict_to_list(pdf_info_dict)
        new_pdf_info_dict = {'pdf_info': pdf_info_list}
        clean_memory(get_device())
        
        return new_pdf_info_dict
    
    def _process_groups(self, groups: List, body_key: str,
                       caption_key: str, footnote_key: str) -> tuple:
        """处理图片或表格组。

        Args:
            groups (List): 组列表
            body_key (str): 主体键名
            caption_key (str): 标题键名
            footnote_key (str): 脚注键名

        Returns:
            tuple: (主体块列表, 标题块列表, 脚注块列表)
        """
        body_blocks = []
        caption_blocks = []
        footnote_blocks = []
        
        for group in groups:
            if body_key in group:
                body_blocks.extend(group[body_key])
            if caption_key in group:
                caption_blocks.extend(group[caption_key])
            if footnote_key in group:
                footnote_blocks.extend(group[footnote_key])
        
        return body_blocks, caption_blocks, footnote_blocks
    
    def _process_spans(self, spans: List, all_bboxes: List,
                      all_discarded_blocks: List) -> List:
        """处理spans信息。

        Args:
            spans (List): spans列表
            all_bboxes (List): 所有边界框
            all_discarded_blocks (List): 所有丢弃的块

        Returns:
            List: 处理后的spans列表
        """
        # 删除重叠的spans
        spans, _ = remove_overlaps_low_confidence_spans(spans)
        spans, _ = remove_overlaps_min_spans(spans)
        
        # 根据解析模式处理spans
        if self.parse_mode == SupportedPdfParseMethod.TXT:
            from magic_pdf.pre_proc.txt_spans_extract_v2 import txt_spans_extract_v2
            spans = txt_spans_extract_v2(
                self.dataset[0], spans, all_bboxes,
                all_discarded_blocks, self.lang
            )
        elif self.parse_mode == SupportedPdfParseMethod.OCR:
            pass
        else:
            raise Exception('parse_mode must be txt or ocr')
        
        return spans