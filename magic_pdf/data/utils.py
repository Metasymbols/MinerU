
import fitz
import numpy as np
from loguru import logger
from typing import Dict, List, Optional, Union
from PIL import Image

from magic_pdf.utils.annotations import ImportPIL

# 定义常量
DEFAULT_DPI = 200
MAX_IMAGE_SIZE = 4500  # 图像缩放的最大尺寸限制

@ImportPIL
def fitz_doc_to_image(doc: fitz.Page, dpi: int = DEFAULT_DPI) -> Dict[str, Union[np.ndarray, int]]:
    """将PyMuPDF页面对象转换为图像，并将图像转换为NumPy数组。

    该函数接收一个PyMuPDF页面对象，将其渲染为图像，并返回包含图像数据和尺寸信息的字典。
    如果渲染后的图像尺寸超过预设的最大限制，将使用原始尺寸（不缩放）重新渲染。

    参数:
        doc (fitz.Page): PyMuPDF页面对象
        dpi (int, optional): 图像的DPI值，用于控制渲染质量。默认为200。
    返回:
        Dict[str, Union[np.ndarray, int]]: 包含以下键值的字典：
            - 'img': 图像数据，类型为numpy.ndarray，形状为(height, width, 3)的RGB图像
            - 'width': 图像宽度（像素）
            - 'height': 图像高度（像素）
    示例:
        >>> doc = fitz.open('example.pdf')[0]  # 打开PDF的第一页
        >>> result = fitz_doc_to_image(doc)
        >>> image = result['img']  # 获取图像数组
        >>> width = result['width']  # 获取图像宽度
        >>> height = result['height']  # 获取图像高度
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pm = doc.get_pixmap(matrix=mat, alpha=False)

    # 如果缩放后的尺寸超过限制，则使用原始尺寸重新渲染
    if pm.width > MAX_IMAGE_SIZE or pm.height > MAX_IMAGE_SIZE:
        pm = doc.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

    img = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
    img = np.array(img)
    return {'img': img, 'width': pm.width, 'height': pm.height}

@ImportPIL
def load_images_from_pdf(pdf_bytes: bytes, dpi: int = DEFAULT_DPI, start_page_id: int = 0, end_page_id: Optional[int] = None) -> List[Dict[str, Union[np.ndarray, int, List]]]:
    """将PDF文件转换为图像列表。

    该函数接收PDF文件的字节数据，将其转换为一系列图像。每个图像都包含图像数据和尺寸信息。
    如果渲染后的图像尺寸超过预设的最大限制，将使用原始尺寸（不缩放）重新渲染。

    参数:
        pdf_bytes (bytes): PDF文件的字节数据
        dpi (int, optional): 图像的DPI值，用于控制渲染质量。默认为200。
        start_page_id (int, optional): 起始页码（从0开始）。默认为0。
        end_page_id (Optional[int], optional): 结束页码。如果为None，则处理到最后一页。默认为None。

    返回:
        List[Dict[str, Union[np.ndarray, int, List]]]: 包含图像信息的字典列表，每个字典包含：
            - 'img': 图像数据，类型为numpy.ndarray（对于有效页面）或空列表（对于无效页面）
            - 'width': 图像宽度（像素）
            - 'height': 图像高度（像素）

    示例:
        >>> pdf_bytes = open('example.pdf', 'rb').read()
        >>> images = load_images_from_pdf(pdf_bytes, dpi=300)
        >>> first_page_image = images[0]['img']
        >>> first_page_width = images[0]['width']
    """
    from PIL import Image
    images = []

    try:
        with fitz.open('pdf', pdf_bytes) as doc:
            pdf_page_num = doc.page_count
            
            # 验证并调整结束页码
            end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else pdf_page_num - 1
            if end_page_id > pdf_page_num - 1:
                logger.warning('结束页码超出范围，将使用PDF总页数作为结束页码')
                end_page_id = pdf_page_num - 1

            # 处理每一页
            for index in range(doc.page_count):
                if start_page_id <= index <= end_page_id:
                    page = doc[index]
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pm = page.get_pixmap(matrix=mat, alpha=False)

                    # 如果缩放后的尺寸超过限制，则使用原始尺寸重新渲染
                    if pm.width > MAX_IMAGE_SIZE or pm.height > MAX_IMAGE_SIZE:
                        pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                    img = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
                    img = np.array(img)
                    img_dict = {'img': img, 'width': pm.width, 'height': pm.height}
                else:
                    img_dict = {'img': [], 'width': 0, 'height': 0}

                images.append(img_dict)

    except Exception as e:
        logger.error(f'PDF转换图像时发生错误: {str(e)}')
        raise

    return images
