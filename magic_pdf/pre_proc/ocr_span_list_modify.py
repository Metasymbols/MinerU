
from magic_pdf.config.drop_tag import DropTag
from magic_pdf.config.ocr_content_type import BlockType
from magic_pdf.libs.boxbase import calculate_iou, get_minbox_if_overlap_by_ratio


def remove_overlaps_low_confidence_spans(spans):
    """删除重叠spans中置信度较低的那些。

    遍历所有spans，如果两个span的重叠度（IOU）超过0.9，则删除置信度较低的那个。

    Args:
        spans (list): 需要处理的span列表，每个span包含bbox和score信息。

    Returns:
        tuple: 返回(spans, dropped_spans)，其中spans为保留的span列表，
              dropped_spans为被删除的span列表。
    """
    dropped_spans = []
    #  删除重叠spans中置信度低的的那些
    for span1 in spans:
        for span2 in spans:
            if span1 != span2:
                # span1 或 span2 任何一个都不应该在 dropped_spans 中
                if span1 in dropped_spans or span2 in dropped_spans:
                    continue
                else:
                    if calculate_iou(span1['bbox'], span2['bbox']) > 0.9:
                        if span1['score'] < span2['score']:
                            span_need_remove = span1
                        else:
                            span_need_remove = span2
                        if (
                            span_need_remove is not None
                            and span_need_remove not in dropped_spans
                        ):
                            dropped_spans.append(span_need_remove)

    if len(dropped_spans) > 0:
        for span_need_remove in dropped_spans:
            spans.remove(span_need_remove)
            span_need_remove['tag'] = DropTag.SPAN_OVERLAP

    return spans, dropped_spans


def check_chars_is_overlap_in_span(chars):
    """检查字符是否在span中重叠。

    遍历所有字符对，如果任意两个字符的重叠度（IOU）超过0.35，则认为存在重叠。

    Args:
        chars (list): 需要检查的字符列表，每个字符包含bbox信息。

    Returns:
        bool: 如果存在重叠返回True，否则返回False。
    """
    for i in range(len(chars)):
        for j in range(i + 1, len(chars)):
            if calculate_iou(chars[i]['bbox'], chars[j]['bbox']) > 0.35:
                return True
    return False


def remove_overlaps_min_spans(spans):
    """删除重叠spans中较小的那些。

    遍历所有spans，如果两个span的重叠比例超过0.65，则删除较小的那个。

    Args:
        spans (list): 需要处理的span列表，每个span包含bbox信息。

    Returns:
        tuple: 返回(spans, dropped_spans)，其中spans为保留的span列表，
              dropped_spans为被删除的span列表。
    """
    dropped_spans = []
    #  删除重叠spans中较小的那些
    for span1 in spans:
        for span2 in spans:
            if span1 != span2:
                # span1 或 span2 任何一个都不应该在 dropped_spans 中
                if span1 in dropped_spans or span2 in dropped_spans:
                    continue
                else:
                    overlap_box = get_minbox_if_overlap_by_ratio(span1['bbox'], span2['bbox'], 0.65)
                    if overlap_box is not None:
                        span_need_remove = next((span for span in spans if span['bbox'] == overlap_box), None)
                        if span_need_remove is not None and span_need_remove not in dropped_spans:
                            dropped_spans.append(span_need_remove)
    if len(dropped_spans) > 0:
        for span_need_remove in dropped_spans:
            spans.remove(span_need_remove)
            span_need_remove['tag'] = DropTag.SPAN_OVERLAP

    return spans, dropped_spans


def get_qa_need_list_v2(blocks):
    """从blocks中提取需要进行QA处理的元素列表。

    从输入的blocks中分别提取图片、表格和行间公式的列表。

    Args:
        blocks (list): 输入的block列表，每个block包含type信息。

    Returns:
        tuple: 返回(images, tables, interline_equations)，分别为图片、表格和行间公式的列表。
    """
    # 创建 images, tables, interline_equations, inline_equations 的副本
    images = []
    tables = []
    interline_equations = []

    for block in blocks:
        if block['type'] == BlockType.Image:
            images.append(block)
        elif block['type'] == BlockType.Table:
            tables.append(block)
        elif block['type'] == BlockType.InterlineEquation:
            interline_equations.append(block)
    return images, tables, interline_equations
