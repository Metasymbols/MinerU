import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import List
from loguru import logger

from magic_pdf.config.exceptions import EmptyData, InvalidParams
from magic_pdf.data.data_reader_writer import (FileBasedDataReader,
                                               MultiBucketS3DataReader)
from magic_pdf.data.dataset import ImageDataset, PymuDocDataset
from magic_pdf.utils.office_to_pdf import convert_file_to_pdf, ConvertToPdfError

def read_jsonl(
    s3_path_or_local: str, s3_client: MultiBucketS3DataReader | None = None
) -> list[PymuDocDataset]:
    """读取JSONL文件并返回PymuDocDataset列表。

    本函数可以读取本地或S3存储上的JSONL文件，并将其中每一行转换为PymuDocDataset对象。
    JSONL文件中的每一行必须包含PDF文件的位置信息（file_location或path字段）。

    参数:
        s3_path_or_local (str): JSONL文件的路径，可以是本地文件路径或S3路径（以's3://'开头）
        s3_client (MultiBucketS3DataReader | None, optional): S3客户端对象，支持多bucket操作。
            当读取S3路径时必须提供此参数。默认为None。

    返回:
        list[PymuDocDataset]: 由JSONL文件中每行对应的PDF文件转换而来的PymuDocDataset对象列表

    异常:
        InvalidParams: 当提供的是S3路径但未提供s3_client时抛出
        EmptyData: 当JSONL文件中某行未提供PDF文件位置信息时抛出
        InvalidParams: 当PDF文件位置是S3路径但未提供s3_client时抛出
        json.JSONDecodeError: 当JSONL文件包含无效的JSON格式时抛出
        FileNotFoundError: 当本地文件不存在时抛出
    """
    logger.info(f"Reading JSONL file from {s3_path_or_local}")
    bits_arr = []
    try:
        if s3_path_or_local.startswith('s3://'):
            if s3_client is None:
                raise InvalidParams('s3_client is required when s3_path is provided')
            jsonl_bits = s3_client.read(s3_path_or_local)
            logger.debug(f"Successfully read {len(jsonl_bits)} bytes from S3")
        else:
            if not os.path.exists(s3_path_or_local):
                raise FileNotFoundError(f"Local file not found: {s3_path_or_local}")
            jsonl_bits = FileBasedDataReader('').read(s3_path_or_local)
            logger.debug(f"Successfully read {len(jsonl_bits)} bytes from local file")

        jsonl_d = []
        for line_no, line in enumerate(jsonl_bits.decode().split('\n'), 1):
            if not line.strip():
                continue
            try:
                jsonl_d.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON at line {line_no}: {e}")
                raise

        logger.info(f"Successfully parsed {len(jsonl_d)} JSON lines")
        for d in jsonl_d:
            pdf_path = d.get('file_location', '') or d.get('path', '')
            if len(pdf_path) == 0:
                raise EmptyData('pdf file location is empty')
            logger.debug(f"Processing PDF path: {pdf_path}")
            if pdf_path.startswith('s3://'):
                if s3_client is None:
                    raise InvalidParams('s3_client is required when s3_path is provided')
                bits_arr.append(s3_client.read(pdf_path))
            else:
                if not os.path.exists(pdf_path):
                    raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                bits_arr.append(FileBasedDataReader('').read(pdf_path))

        return [PymuDocDataset(bits) for bits in bits_arr]

    except Exception as e:
        logger.exception(f"Error processing JSONL file: {str(e)}")
        raise

def read_local_pdfs(path: str) -> list[PymuDocDataset]:
    """读取本地PDF文件或目录中的所有PDF文件。

    本函数可以读取单个PDF文件或包含多个PDF文件的目录。当提供目录路径时，
    将递归遍历该目录及其子目录，读取所有扩展名为.pdf的文件。

    参数:
        path (str): PDF文件的路径或包含PDF文件的目录路径

    返回:
        list[PymuDocDataset]: 读取到的所有PDF文件转换而来的PymuDocDataset对象列表

    异常:
        FileNotFoundError: 当提供的路径不存在时抛出
        InvalidParams: 当提供的文件不是PDF文件时抛出
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    logger.info(f"Reading PDFs from {path}")
    reader = FileBasedDataReader()
    ret = []

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            pdf_files = [f for f in files if Path(f).suffix.lower() == '.pdf']
            logger.debug(f"Found {len(pdf_files)} PDF files in {root}")
            for file in pdf_files:
                file_path = os.path.join(root, file)
                try:
                    ret.append(PymuDocDataset(reader.read(file_path)))
                    logger.debug(f"Successfully read PDF: {file_path}")
                except Exception as e:
                    logger.error(f"Error reading PDF {file_path}: {str(e)}")
                    raise
    else:
        if not path.lower().endswith('.pdf'):
            raise InvalidParams(f"File is not a PDF: {path}")
        try:
            ret.append(PymuDocDataset(reader.read(path)))
            logger.debug(f"Successfully read PDF: {path}")
        except Exception as e:
            logger.error(f"Error reading PDF {path}: {str(e)}")
            raise

    logger.info(f"Successfully read {len(ret)} PDF files")
    return ret

def read_local_office(path: str) -> list[PymuDocDataset]:
    """读取本地Office文件或目录中的所有Office文件。

    本函数支持读取Microsoft Office文档（支持的格式：ppt、pptx、doc、docx），
    可以处理单个文件或包含多个文件的目录。所有Office文件都会被转换为PDF格式后再处理。
    转换过程使用临时目录，处理完成后会自动清理。

    参数:
        path (str): Office文件的路径或包含Office文件的目录路径

    返回:
        list[PymuDocDataset]: 所有Office文件转换为PDF后的PymuDocDataset对象列表

    异常:
        ConvertToPdfError: 当使用libreoffice转换Office文件到PDF格式失败时抛出
        FileNotFoundError: 当提供的路径不存在时抛出
        InvalidParams: 当文件扩展名不是支持的Office格式时抛出
        Exception: 其他未知异常
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    logger.info(f"Reading Office files from {path}")
    suffixes = {'.ppt', '.pptx', '.doc', '.docx'}
    fns = []
    ret = []

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            office_files = [f for f in files if Path(f).suffix.lower() in suffixes]
            logger.debug(f"Found {len(office_files)} Office files in {root}")
            for file in office_files:
                fns.append(os.path.join(root, file))
    else:
        if Path(path).suffix.lower() not in suffixes:
            raise InvalidParams(f"Unsupported file type: {path}")
        fns.append(path)

    reader = FileBasedDataReader()
    temp_dir = tempfile.mkdtemp(prefix='office_convert_')
    logger.debug(f"Created temporary directory: {temp_dir}")

    try:
        for fn in fns:
            try:
                logger.debug(f"Converting {fn} to PDF")
                convert_file_to_pdf(fn, temp_dir)
                fn_path = Path(fn)
                pdf_fn = f"{temp_dir}/{fn_path.stem}.pdf"
                ret.append(PymuDocDataset(reader.read(pdf_fn)))
                logger.debug(f"Successfully converted and read: {fn}")
            except Exception as e:
                logger.error(f"Error processing file {fn}: {str(e)}")
                raise
    finally:
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory {temp_dir}: {str(e)}")

    logger.info(f"Successfully processed {len(ret)} Office files")
    return ret

def read_local_images(path: str, suffixes: list[str]=['.png', '.jpg']) -> list[ImageDataset]:
    """读取本地图片文件或目录中的所有图片文件。

    本函数可以读取单个图片文件或包含多个图片文件的目录。当提供目录路径时，
    将递归遍历该目录及其子目录，读取所有扩展名匹配的图片文件。

    参数:
        path (str): 图片文件的路径或包含图片文件的目录路径
        suffixes (list[str]): 要处理的图片文件扩展名列表，用于过滤文件。
            默认支持'.png'和'.jpg'格式。例如：['.jpg', '.png']

    返回:
        list[ImageDataset]: 读取到的所有图片文件转换而来的ImageDataset对象列表

    异常:
        FileNotFoundError: 当提供的路径不存在时抛出
        InvalidParams: 当文件扩展名不在指定的suffixes列表中时抛出
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    logger.info(f"Reading images from {path} with suffixes {suffixes}")
    s_suffixes = {suffix.lower() for suffix in suffixes}
    reader = FileBasedDataReader()
    imgs_bits = []

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            image_files = [f for f in files if Path(f).suffix.lower() in s_suffixes]
            logger.debug(f"Found {len(image_files)} image files in {root}")
            for file in image_files:
                file_path = os.path.join(root, file)
                try:
                    imgs_bits.append(reader.read(file_path))
                    logger.debug(f"Successfully read image: {file_path}")
                except Exception as e:
                    logger.error(f"Error reading image {file_path}: {str(e)}")
                    raise
    else:
        if Path(path).suffix.lower() not in s_suffixes:
            raise InvalidParams(f"Unsupported image type: {path}")
        try:
            imgs_bits.append(reader.read(path))
            logger.debug(f"Successfully read image: {path}")
        except Exception as e:
            logger.error(f"Error reading image {path}: {str(e)}")
            raise

    logger.info(f"Successfully read {len(imgs_bits)} image files")
    return [ImageDataset(bits) for bits in imgs_bits]
