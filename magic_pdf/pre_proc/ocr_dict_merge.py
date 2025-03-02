from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.libs.boxbase import __is_overlaps_y_exceeds_threshold, calculate_overlap_area_in_bbox1_area_ratio


# 将每一个line中的span从左到右排序
def line_sort_spans_by_left_to_right(lines):
    """将每一行中的span按照从左到右的顺序排序。

    Args:
        lines (list): 包含多个span的行列表，每个span包含bbox信息。

    Returns:
        list: 返回排序后的行对象列表，每个行对象包含bbox和spans信息。
    """
    line_objects = []
    for line in lines:
        #  按照x0坐标排序
        line.sort(key=lambda span: span['bbox'][0])
        line_bbox = [
            min(span['bbox'][0] for span in line),  # x0
            min(span['bbox'][1] for span in line),  # y0
            max(span['bbox'][2] for span in line),  # x1
            max(span['bbox'][3] for span in line),  # y1
        ]
        line_objects.append({
            'bbox': line_bbox,
            'spans': line,
        })
    return line_objects


def merge_spans_to_line(spans, threshold=0.6):
    """将spans合并成行。

    根据spans的垂直位置关系将其合并成行。如果span是行间公式、图片或表格，或者当前行已包含这些类型，
    则会开始新行。如果span与当前行的最后一个span在y轴上的重叠超过阈值，则将其添加到当前行。

    Args:
        spans (list): 需要合并的span列表。
        threshold (float, optional): y轴重叠的阈值。默认为0.6。

    Returns:
        list: 合并后的行列表，每行包含多个span。
    """
    if len(spans) == 0:
        return []
    else:
        # 按照y0坐标排序
        spans.sort(key=lambda span: span['bbox'][1])

        lines = []
        current_line = [spans[0]]
        for span in spans[1:]:
            # 如果当前的span类型为"interline_equation" 或者 当前行中已经有"interline_equation"
            # image和table类型，同上
            if span['type'] in [
                    ContentType.InterlineEquation, ContentType.Image,
                    ContentType.Table
            ] or any(s['type'] in [
                    ContentType.InterlineEquation, ContentType.Image,
                    ContentType.Table
            ] for s in current_line):
                # 则开始新行
                lines.append(current_line)
                current_line = [span]
                continue

            # 如果当前的span与当前行的最后一个span在y轴上重叠，则添加到当前行
            if __is_overlaps_y_exceeds_threshold(span['bbox'], current_line[-1]['bbox'], threshold):
                current_line.append(span)
            else:
                # 否则，开始新行
                lines.append(current_line)
                current_line = [span]

        # 添加最后一行
        if current_line:
            lines.append(current_line)

        return lines


def span_block_type_compatible(span_type, block_type):
    if span_type in [ContentType.Text, ContentType.InlineEquation]:
        return block_type in [BlockType.Text, BlockType.Title, BlockType.ImageCaption, BlockType.ImageFootnote, BlockType.TableCaption, BlockType.TableFootnote]
    elif span_type == ContentType.InterlineEquation:
        return block_type in [BlockType.InterlineEquation]
    elif span_type == ContentType.Image:
        return block_type in [BlockType.ImageBody]
    elif span_type == ContentType.Table:
        return block_type in [BlockType.TableBody]
    else:
        return False


def fill_spans_in_blocks(blocks, spans, radio):
    """将spans按位置关系分配到对应的blocks中。

    根据span与block的重叠面积比例，将span分配到相应的block中。对于图片和表格类型的block，
    还会记录group_id信息。

    Args:
        blocks (list): block列表，每个block包含类型和位置信息。
        spans (list): 需要分配的span列表。
        radio (float): 重叠面积比例的阈值。

    Returns:
        tuple: 返回(block_with_spans, spans)，其中block_with_spans为包含spans的block列表，
              spans为未被分配的span列表。
    """
    """将allspans中的span按位置关系，放入blocks中."""
    block_with_spans = []
    for block in blocks:
        block_type = block[7]
        block_bbox = block[0:4]
        block_dict = {
            'type': block_type,
            'bbox': block_bbox,
        }
        if block_type in [
            BlockType.ImageBody, BlockType.ImageCaption, BlockType.ImageFootnote,
            BlockType.TableBody, BlockType.TableCaption, BlockType.TableFootnote
        ]:
            block_dict['group_id'] = block[-1]
        block_spans = []
        for span in spans:
            span_bbox = span['bbox']
            if calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > radio and span_block_type_compatible(span['type'], block_type):
                block_spans.append(span)

        block_dict['spans'] = block_spans
        block_with_spans.append(block_dict)

        # 从spans删除已经放入block_spans中的span
        if len(block_spans) > 0:
            for span in block_spans:
                spans.remove(span)

    return block_with_spans, spans


def fix_block_spans_v2(block_with_spans):
    """修复和处理包含spans的blocks。

    根据block的类型进行不同的处理：
    - 对于文本类block（文本、标题、图片说明、表格说明等），调用fix_text_block处理
    - 对于特殊类型block（行间公式、图片体、表格体），调用fix_interline_block处理

    Args:
        block_with_spans (list): 包含spans的block列表。

    Returns:
        list: 处理后的block列表。
    """
    fix_blocks = []
    for block in block_with_spans:
        block_type = block['type']

        if block_type in [BlockType.Text, BlockType.Title,
                          BlockType.ImageCaption, BlockType.ImageFootnote,
                          BlockType.TableCaption, BlockType.TableFootnote
                          ]:
            block = fix_text_block(block)
        elif block_type in [BlockType.InterlineEquation, BlockType.ImageBody, BlockType.TableBody]:
            block = fix_interline_block(block)
        else:
            continue
        fix_blocks.append(block)
    return fix_blocks


def fix_discarded_block(discarded_block_with_spans):
    """修复被丢弃的blocks。

    对被丢弃的blocks进行文本处理，将其中的spans合并成行。

    Args:
        discarded_block_with_spans (list): 包含spans的被丢弃block列表。

    Returns:
        list: 处理后的被丢弃block列表。
    """
    fix_discarded_blocks = []
    for block in discarded_block_with_spans:
        block = fix_text_block(block)
        fix_discarded_blocks.append(block)
    return fix_discarded_blocks


def fix_text_block(block):
    """修复文本类型的block。

    将block中的行间公式类型转换为行内公式类型，并将spans合并成行。

    Args:
        block (dict): 包含spans的文本block。

    Returns:
        dict: 处理后的block，包含合并后的行信息。
    """
    # 文本block中的公式span都应该转换成行内type
    for span in block['spans']:
        if span['type'] == ContentType.InterlineEquation:
            span['type'] = ContentType.InlineEquation
    block_lines = merge_spans_to_line(block['spans'])
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    block['lines'] = sort_block_lines
    del block['spans']
    return block


def fix_interline_block(block):
    """修复特殊类型的block（行间公式、图片体、表格体）。

    将block中的spans合并成行并按从左到右排序。

    Args:
        block (dict): 包含spans的特殊类型block。

    Returns:
        dict: 处理后的block，包含合并后的行信息。
    """
    block_lines = merge_spans_to_line(block['spans'])
    sort_block_lines = line_sort_spans_by_left_to_right(block_lines)
    block['lines'] = sort_block_lines
    del block['spans']
    return block
