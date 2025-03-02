import time

import torch
from PIL import Image
from loguru import logger

from magic_pdf.libs.clean_memory import clean_memory


def crop_img(input_res, input_pil_img, crop_paste_x=0, crop_paste_y=0):
    """裁剪图像并将其粘贴到白色背景上。

    Args:
        input_res (dict): 包含裁剪区域坐标的字典，必须包含'poly'键，其值为[xmin,ymin,_,_,xmax,ymax]格式的列表。
        input_pil_img (PIL.Image): 输入的PIL图像对象。
        crop_paste_x (int, optional): 水平方向的粘贴偏移量。默认为0。
        crop_paste_y (int, optional): 垂直方向的粘贴偏移量。默认为0。

    Returns:
        tuple: 包含以下元素的元组：
            - PIL.Image: 裁剪并粘贴后的图像
            - list: [paste_x, paste_y, xmin, ymin, xmax, ymax, new_width, new_height]格式的坐标信息
    """
    crop_xmin, crop_ymin = int(input_res['poly'][0]), int(input_res['poly'][1])
    crop_xmax, crop_ymax = int(input_res['poly'][4]), int(input_res['poly'][5])
    # Create a white background with an additional width and height of 50
    crop_new_width = crop_xmax - crop_xmin + crop_paste_x * 2
    crop_new_height = crop_ymax - crop_ymin + crop_paste_y * 2
    return_image = Image.new('RGB', (crop_new_width, crop_new_height), 'white')

    # Crop image
    crop_box = (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
    cropped_img = input_pil_img.crop(crop_box)
    return_image.paste(cropped_img, (crop_paste_x, crop_paste_y))
    return_list = [crop_paste_x, crop_paste_y, crop_xmin, crop_ymin, crop_xmax, crop_ymax, crop_new_width, crop_new_height]
    return return_image, return_list


# Select regions for OCR / formula regions / table regions
def get_res_list_from_layout_res(layout_res):
    """从布局检测结果中提取不同类型区域的列表。

    Args:
        layout_res (list): 布局检测结果列表，每个元素为包含'category_id'和'poly'的字典。

    Returns:
        tuple: 包含以下三个列表的元组：
            - list: OCR区域列表
            - list: 表格区域列表
            - list: 公式区域列表
    """
    ocr_res_list = []
    table_res_list = []
    single_page_mfdetrec_res = []
    for res in layout_res:
        if int(res['category_id']) in [13, 14]:
            single_page_mfdetrec_res.append({
                "bbox": [int(res['poly'][0]), int(res['poly'][1]),
                         int(res['poly'][4]), int(res['poly'][5])],
            })
        elif int(res['category_id']) in [0, 1, 2, 4, 6, 7]:
            ocr_res_list.append(res)
        elif int(res['category_id']) in [5]:
            table_res_list.append(res)
    return ocr_res_list, table_res_list, single_page_mfdetrec_res


def clean_vram(device, vram_threshold=8):
    """根据显存阈值清理设备内存。

    当设备可用显存小于等于阈值时，执行内存清理操作。

    Args:
        device (str): 设备标识符，如'cuda'、'npu'等。
        vram_threshold (int, optional): 显存阈值，单位为GB。默认为8GB。
    """
    total_memory = get_vram(device)
    if total_memory and total_memory <= vram_threshold:
        gc_start = time.time()
        clean_memory(device)
        gc_time = round(time.time() - gc_start, 2)
        logger.info(f"gc time: {gc_time}")


def get_vram(device):
    """获取指定设备的显存大小。

    Args:
        device (str): 设备标识符，如'cuda'、'npu'等。

    Returns:
        float or None: 设备的显存大小（单位：GB），如果设备不可用则返回None。
    """
    if torch.cuda.is_available() and device != 'cpu':
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # 将字节转换为 GB
        return total_memory
    elif str(device).startswith("npu"):
        import torch_npu
        if torch_npu.npu.is_available():
            total_memory = torch_npu.npu.get_device_properties(device).total_memory / (1024 ** 3)  # 转为 GB
            return total_memory
    else:
        return None