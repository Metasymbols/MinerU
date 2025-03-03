from typing import Dict, List, Optional
from .base import PDFParser
from magic_pdf.config.ocr_content_type import BlockType
from magic_pdf.libs.pdf_image_tools import extract_image, process_image


class ImageProcessor(PDFParser):
    """图像处理器，负责处理PDF文档中的图像内容。
    
    此类主要负责：
    1. 图像内容的提取
    2. 图像处理和优化
    3. 图像质量控制
    4. 图像格式转换
    
    继承自PDFParser基类，实现了图像处理相关的具体功能。
    """
    
    def parse_page(self, page_id: int) -> Dict:
        """解析单个页面的图像内容。

        Args:
            page_id (int): 页面ID

        Returns:
            Dict: 页面图像解析结果
        """
        page = self.dataset[page_id]
        page_info = page.get_page_info()
        
        # 获取页面图像内容
        image_blocks = self.magic_model.get_image_blocks(page_id)
        processed_blocks = self._process_image_blocks(image_blocks)
        
        return {
            'page_id': page_id,
            'page_w': page_info.w,
            'page_h': page_info.h,
            'image_blocks': processed_blocks
        }
    
    def parse_document(self, start_page_id: int = 0,
                      end_page_id: Optional[int] = None,
                      debug_mode: bool = False) -> Dict:
        """解析整个文档的图像内容。

        Args:
            start_page_id (int, optional): 起始页码. Defaults to 0.
            end_page_id (Optional[int], optional): 结束页码. Defaults to None.
            debug_mode (bool, optional): 调试模式. Defaults to False.

        Returns:
            Dict: 文档图像解析结果
        """
        end_page_id = self._validate_page_range(start_page_id, end_page_id)
        image_results = {}
        
        for page_id in range(start_page_id, end_page_id + 1):
            image_results[f'page_{page_id}'] = self.parse_page(page_id)
        
        return image_results
    
    def _process_image_blocks(self, blocks: List) -> List:
        """处理图像块。

        Args:
            blocks (List): 图像块列表

        Returns:
            List: 处理后的图像块列表
        """
        processed_blocks = []
        
        for block in blocks:
            if block.type == BlockType.IMAGE:
                # 提取图像
                image_data = extract_image(block.image)
                # 处理图像
                processed_image = process_image(image_data)
                block.image = processed_image
            processed_blocks.append(block)
        
        return processed_blocks