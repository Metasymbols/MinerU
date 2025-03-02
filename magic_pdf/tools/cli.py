import os
import shutil
import tempfile
import click
import fitz
from loguru import logger
from pathlib import Path

import magic_pdf.model as model_config
from magic_pdf.data.data_reader_writer import FileBasedDataReader
from magic_pdf.libs.version import __version__
from magic_pdf.tools.common import do_parse, parse_pdf_methods
from magic_pdf.utils.office_to_pdf import convert_file_to_pdf

pdf_suffixes = ['.pdf']
ms_office_suffixes = ['.ppt', '.pptx', '.doc', '.docx']
image_suffixes = ['.png', '.jpeg', '.jpg']


@click.command()
@click.version_option(__version__,
                      '--version',
                      '-v',
                      help='显示版本信息并退出')
@click.option(
    '-p',
    '--path',
    'path',
    type=click.Path(exists=True),
    required=True,
    help='本地文件路径或目录。支持的文件格式：PDF、PPT、PPTX、DOC、DOCX、PNG、JPG',
)
@click.option(
    '-o',
    '--output-dir',
    'output_dir',
    type=click.Path(),
    required=True,
    help='输出结果的本地目录路径',
)
@click.option(
    '-m',
    '--method',
    'method',
    type=parse_pdf_methods,
    help="""the method for parsing pdf.
ocr: using ocr technique to extract information from pdf.
txt: suitable for the text-based pdf only and outperform ocr.
auto: automatically choose the best method for parsing pdf from ocr and txt.
without method specified, auto will be used by default.""",
    default='auto',
)
@click.option(
    '-l',
    '--lang',
    'lang',
    type=str,
    help="""
    输入PDF文档中的语言（如果已知）以提高OCR准确性。可选参数。
    您需要输入语言的"缩写代码"，支持的语言列表请参考：
    https://paddlepaddle.github.io/PaddleOCR/latest/en/ppocr/blog/multi_languages.html#5-support-languages-and-abbreviations
    """,
    default=None,
)
@click.option(
    '-d',
    '--debug',
    'debug_able',
    type=bool,
    help='在执行CLI命令期间启用详细的调试信息输出。',
    default=False,
)
@click.option(
    '-s',
    '--start',
    'start_page_id',
    type=int,
    help='PDF解析的起始页码（从0开始计数）。',
    default=0,
)
@click.option(
    '-e',
    '--end',
    'end_page_id',
    type=int,
    help='PDF解析的结束页码（从0开始计数）。',
    default=None,
)
def cli(path, output_dir, method, lang, debug_able, start_page_id, end_page_id):
    model_config.__use_inside_model__ = True
    model_config.__model_mode__ = 'full'
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    def read_fn(path: Path):
        if path.suffix in ms_office_suffixes:
            convert_file_to_pdf(str(path), temp_dir)
            fn = os.path.join(temp_dir, f"{path.stem}.pdf")
        elif path.suffix in image_suffixes:
            with open(str(path), 'rb') as f:
                bits = f.read()
            pdf_bytes = fitz.open(stream=bits).convert_to_pdf()
            fn = os.path.join(temp_dir, f"{path.stem}.pdf")
            with open(fn, 'wb') as f:
                f.write(pdf_bytes)
        elif path.suffix in pdf_suffixes:
            fn = str(path)
        else:
            raise Exception(f"Unknown file suffix: {path.suffix}")
        
        disk_rw = FileBasedDataReader(os.path.dirname(fn))
        return disk_rw.read(os.path.basename(fn))

    def parse_doc(doc_path: Path):
        try:
            file_name = str(Path(doc_path).stem)
            pdf_data = read_fn(doc_path)
            do_parse(
                output_dir,
                file_name,
                pdf_data,
                [],
                method,
                debug_able,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                lang=lang
            )

        except Exception as e:
            logger.exception(e)

    if os.path.isdir(path):
        for doc_path in Path(path).glob('*'):
            if doc_path.suffix in pdf_suffixes + image_suffixes + ms_office_suffixes:
                parse_doc(doc_path)
    else:
        parse_doc(Path(path))

    shutil.rmtree(temp_dir)


if __name__ == '__main__':
    cli()
