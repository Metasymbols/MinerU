from typing import Dict, List, Optional

from magic_pdf.config.ocr_content_type import BlockType
from magic_pdf.pre_proc.ocr_detect_all_bboxes import ocr_prepare_bboxes_for_layout_split_v2
from magic_pdf.pre_proc.ocr_dict_merge import fill_spans_in_blocks, fix_block_spans_v2, fix_discarded_block
from magic_pdf.pre_proc.ocr_span_list_modify import get_qa_need_list_v2
from .base import PDFParser


class LayoutParser(PDFParser):
    """PDF布局解析器，负责处理文档的布局结构。
    
    此类主要处理PDF文档的布局分析，包括：
    1. 提取页面基本信息和区块
    2. 处理图片和表格区块
    3. 整理区块边界框
    4. 处理特殊块类型
    
    继承自PDFParser基类，实现了布局分析相关的具体功能。
    """
    
    def parse_page(self, page_id: int) -> Dict:
        """解析单个页面的布局。

        Args:
            page_id (int): 页面ID

        Returns:
            Dict: 页面布局解析结果
        """
        # 获取页面基本信息和区块
        page_w, page_h = self.magic_model.get_page_size(page_id)
        img_groups = self.magic_model.get_imgs_v2(page_id)
        table_groups = self.magic_model.get_tables_v2(page_id)
        discarded_blocks = self.magic_model.get_discarded(page_id)
        text_blocks = self.magic_model.get_text_blocks(page_id)
        title_blocks = self.magic_model.get_title_blocks(page_id)
        _, _, interline_equations = self.magic_model.get_equations(page_id)
        
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
        
        return {
            'page_w': page_w,
            'page_h': page_h,
            'all_bboxes': all_bboxes,
            'all_discarded_blocks': all_discarded_blocks
        }
    
    def parse_document(self, start_page_id: int = 0,
                      end_page_id: Optional[int] = None,
                      debug_mode: bool = False) -> Dict:
        """解析整个文档的布局。

        Args:
            start_page_id (int, optional): 起始页码. Defaults to 0.
            end_page_id (Optional[int], optional): 结束页码. Defaults to None.
            debug_mode (bool, optional): 调试模式. Defaults to False.

        Returns:
            Dict: 文档布局解析结果
        """
        end_page_id = self._validate_page_range(start_page_id, end_page_id)
        layout_results = {}
        
        for page_id in range(start_page_id, end_page_id + 1):
            layout_results[f'page_{page_id}'] = self.parse_page(page_id)
        
        return layout_results
    
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