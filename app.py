# Copyright (c) Opendatalab. All rights reserved.

import base64
import os
import time
import zipfile
from pathlib import Path
import re

from loguru import logger

from magic_pdf.libs.hash_utils import compute_sha256
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.tools.common import do_parse, prepare_env

# os.system("pip install gradio")
# os.system("pip install gradio-pdf")
import gradio as gr
from gradio_pdf import PDF


def read_fn(path):
    disk_rw = DiskReaderWriter(os.path.dirname(path))
    return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)


def parse_pdf(doc_path, output_dir, end_page_id):
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_name = f"{str(Path(doc_path).stem)}_{time.time()}"
        pdf_data = read_fn(doc_path)
        parse_method = "auto"
        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method)
        do_parse(
            output_dir,
            file_name,
            pdf_data,
            [],
            parse_method,
            False,
            end_page_id=end_page_id,
        )
        return local_md_dir, file_name
    except Exception as e:
        logger.exception(e)


def compress_directory_to_zip(directory_path, output_zip_path):
    """
    压缩指定目录到一个 ZIP 文件。

    :param directory_path: 要压缩的目录路径
    :param output_zip_path: 输出的 ZIP 文件路径
    """
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

            # 遍历目录中的所有文件和子目录
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    # 构建完整的文件路径
                    file_path = os.path.join(root, file)
                    # 计算相对路径
                    arcname = os.path.relpath(file_path, directory_path)
                    # 添加文件到 ZIP 文件
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def replace_image_with_base64(markdown_text, image_dir_path):
    # 匹配Markdown中的图片标签
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'

    # 替换图片链接
    def replace(match):
        relative_path = match.group(1)
        full_path = os.path.join(image_dir_path, relative_path)
        base64_image = image_to_base64(full_path)
        return f"![{relative_path}](data:image/jpeg;base64,{base64_image})"

    # 应用替换
    return re.sub(pattern, replace, markdown_text)


def to_markdown(file_path, end_pages):
    # 获取识别的md文件以及压缩包文件路径
    local_md_dir, file_name = parse_pdf(file_path, './output', end_pages - 1)
    archive_zip_path = os.path.join("./output", compute_sha256(local_md_dir) + ".zip")
    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    if zip_archive_success == 0:
        logger.info("压缩成功")
    else:
        logger.error("压缩失败")
    md_path = os.path.join(local_md_dir, file_name + ".md")
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    md_content = replace_image_with_base64(txt_content, local_md_dir)
    # 返回转换后的PDF路径
    new_pdf_path = os.path.join(local_md_dir, file_name + "_layout.pdf")

    return md_content, txt_content, archive_zip_path, new_pdf_path


# def show_pdf(file_path):
#     with open(file_path, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" ' \
#                   f'width="100%" height="1000" type="application/pdf">'
#     return pdf_display


latex_delimiters = [{"left": "$$", "right": "$$", "display": True},
                    {"left": '$', "right": '$', "display": False}]


def init_model():
    from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
    try:
        model_manager = ModelSingleton()
        txt_model = model_manager.get_model(False, False)
        logger.info(f"txt_model init final")
        ocr_model = model_manager.get_model(True, False)
        logger.info(f"ocr_model init final")
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


model_init = init_model()
logger.info(f"model_init: {model_init}")


# 如果当前模块是主模块，则执行以下代码
if __name__ == "__main__":
    # 创建一个gr.Blocks对象，命名为demo
    with gr.Blocks() as demo:
        # 创建一个gr.Row对象，命名为bu_flow
        with gr.Row():
            # 创建一个gr.Column对象，命名为pdf_show，设置其样式为panel，缩放比例为5
            with gr.Column(variant='panel', scale=5):
                # 创建一个gr.Markdown对象，命名为pdf_show
                pdf_show = gr.Markdown()
                # 创建一个gr.Slider对象，命名为max_pages，设置其最小值为1，最大值为10，默认值为5，步长为1，标签为Max convert pages
                max_pages = gr.Slider(1, 500, 5, step=1, label="Max convert pages")
                # 创建一个gr.Row对象，命名为bu_flow
                with gr.Row() as bu_flow:
                    # 创建一个gr.Button对象，命名为change_bu，标签为Convert
                    change_bu = gr.Button("Convert")
                    # 创建一个gr.ClearButton对象，命名为clear_bu，设置其输入为pdf_show，默认值为Clear
                    clear_bu = gr.ClearButton([pdf_show], value="Clear")
                # 创建一个PDF对象，命名为pdf_show，设置其标签为Please upload pdf，交互式为True，高度为800
                pdf_show = PDF(label="Please upload pdf", interactive=True, height=800)

            # 创建一个gr.Column对象，命名为output_file，设置其样式为panel，缩放比例为5
            with gr.Column(variant='panel', scale=5):
                # 创建一个gr.File对象，命名为output_file，设置其标签为convert result，交互式为False
                output_file = gr.File(label="convert result", interactive=False)
                # 创建一个gr.Tabs对象
                with gr.Tabs():
                    # 创建一个gr.Tab对象，命名为Markdown rendering
                    with gr.Tab("Markdown rendering"):
                        # 创建一个gr.Markdown对象，命名为md，设置其标签为Markdown rendering，高度为900，显示复制按钮，设置latex_delimiters和line_breaks参数
                        md = gr.Markdown(label="Markdown rendering", height=900, show_copy_button=True,
                                         latex_delimiters=latex_delimiters, line_breaks=True)
                    # 创建一个gr.Tab对象，命名为Markdown text
                    with gr.Tab("Markdown text"):
                        # 创建一个gr.TextArea对象，命名为md_text，设置其行数为45，显示复制按钮
                        md_text = gr.TextArea(lines=45, show_copy_button=True)
        # 当change_bu被点击时，执行to_markdown函数，输入为pdf_show和max_pages，输出为md、md_text、output_file和pdf_show
        change_bu.click(fn=to_markdown, inputs=[pdf_show, max_pages], outputs=[md, md_text, output_file, pdf_show])
        # 当clear_bu被点击时，执行clear_bu.add函数，输入为md、pdf_show、md_text和output_file
        clear_bu.add([md, pdf_show, md_text, output_file])

    # 启动demo，设置share参数为True
    demo.launch(share=True)

