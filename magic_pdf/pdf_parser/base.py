from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from magic_pdf.data.dataset import Dataset
from magic_pdf.model.magic_model import MagicModel


class PDFParser(ABC):
    """PDF解析器基类，定义PDF解析的基本接口。
    
    此类作为所有PDF解析器的基类，定义了必要的接口和通用功能。
    继承此类的解析器必须实现parse_page和parse_document方法。
    
    Attributes:
        dataset (Dataset): PDF数据集对象
        magic_model (MagicModel): 魔法模型对象
        image_writer: 图像写入器
        parse_mode (str): 解析模式
        lang (str): 语言代码
    """
    
    def __init__(self, dataset: Dataset, magic_model: MagicModel, image_writer,
                 parse_mode: str, lang: Optional[str] = None):
        """初始化PDF解析器。

        Args:
            dataset (Dataset): PDF数据集对象
            magic_model (MagicModel): 魔法模型对象
            image_writer: 图像写入器
            parse_mode (str): 解析模式
            lang (Optional[str], optional): 语言代码. Defaults to None.
        """
        self.dataset = dataset
        self.magic_model = magic_model
        self.image_writer = image_writer
        self.parse_mode = parse_mode
        self.lang = lang
    
    @abstractmethod
    def parse_page(self, page_id: int) -> Dict:
        """解析单个页面。

        Args:
            page_id (int): 页面ID

        Returns:
            Dict: 页面解析结果
        """
        pass
    
    @abstractmethod
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
        pass
    
    def _validate_page_range(self, start_page_id: int,
                           end_page_id: Optional[int] = None) -> int:
        """验证页面范围的有效性。

        Args:
            start_page_id (int): 起始页码
            end_page_id (Optional[int], optional): 结束页码. Defaults to None.

        Returns:
            int: 有效的结束页码
        """
        if end_page_id is None or end_page_id < 0:
            end_page_id = len(self.dataset) - 1
        if end_page_id >= len(self.dataset):
            end_page_id = len(self.dataset) - 1
        return end_page_id