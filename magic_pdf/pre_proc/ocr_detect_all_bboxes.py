from typing import List, Tuple, Optional, Set

from magic_pdf.config.ocr_content_type import BlockType
from magic_pdf.libs.boxbase import (
    calculate_iou,
    calculate_overlap_area_in_bbox1_area_ratio,
    calculate_vertical_projection_overlap_ratio,
    get_minbox_if_overlap_by_ratio
)
from magic_pdf.pre_proc.remove_bbox_overlap import remove_overlap_between_bbox_for_block


def add_bboxes(blocks: List[dict], block_type: BlockType, bboxes: List[list]) -> None:
    """将blocks中的边界框信息添加到bboxes列表中。

    根据block类型的不同，添加不同格式的边界框信息。对于图片和表格类型的block，
    会额外包含group_id信息。

    Args:
        blocks (list): 包含边界框信息的block列表。
        block_type (BlockType): block的类型。
        bboxes (list): 用于存储边界框信息的列表。
    """
    for block in blocks:
        x0, y0, x1, y1 = block['bbox']
        if block_type in [
            BlockType.ImageBody,
            BlockType.ImageCaption,
            BlockType.ImageFootnote,
            BlockType.TableBody,
            BlockType.TableCaption,
            BlockType.TableFootnote,
        ]:
            bboxes.append(
                [
                    x0,
                    y0,
                    x1,
                    y1,
                    None,
                    None,
                    None,
                    block_type,
                    None,
                    None,
                    None,
                    None,
                    block['score'],
                    block['group_id'],
                ]
            )
        else:
            bboxes.append(
                [
                    x0,
                    y0,
                    x1,
                    y1,
                    None,
                    None,
                    None,
                    block_type,
                    None,
                    None,
                    None,
                    None,
                    block['score'],
                ]
            )


def ocr_prepare_bboxes_for_layout_split_v2(
    img_body_blocks: List[dict],
    img_caption_blocks: List[dict],
    img_footnote_blocks: List[dict],
    table_body_blocks: List[dict],
    table_caption_blocks: List[dict],
    table_footnote_blocks: List[dict],
    discarded_blocks: List[dict],
    text_blocks: List[dict],
    title_blocks: List[dict],
    interline_equation_blocks: List[dict],
    page_w: float,
    page_h: float
) -> Tuple[List[list], List[list]]:
    """准备用于布局分割的边界框信息。

    处理各种类型的blocks（图片、表格、文本等），解决block嵌套问题，
    处理重叠情况，识别和处理脚注，最后对所有边界框进行排序。

    Args:
        img_body_blocks (list): 图片主体blocks。
        img_caption_blocks (list): 图片标题blocks。
        img_footnote_blocks (list): 图片脚注blocks。
        table_body_blocks (list): 表格主体blocks。
        table_caption_blocks (list): 表格标题blocks。
        table_footnote_blocks (list): 表格脚注blocks。
        discarded_blocks (list): 被丢弃的blocks。
        text_blocks (list): 文本blocks。
        title_blocks (list): 标题blocks。
        interline_equation_blocks (list): 行间公式blocks。
        page_w (float): 页面宽度。
        page_h (float): 页面高度。

    Returns:
        tuple: 返回(all_bboxes, all_discarded_blocks)，其中all_bboxes为处理后的边界框列表，
              all_discarded_blocks为被丢弃的边界框列表。
    """
   
    all_bboxes = []

    add_bboxes(img_body_blocks, BlockType.ImageBody, all_bboxes)
    add_bboxes(img_caption_blocks, BlockType.ImageCaption, all_bboxes)
    add_bboxes(img_footnote_blocks, BlockType.ImageFootnote, all_bboxes)
    add_bboxes(table_body_blocks, BlockType.TableBody, all_bboxes)
    add_bboxes(table_caption_blocks, BlockType.TableCaption, all_bboxes)
    add_bboxes(table_footnote_blocks, BlockType.TableFootnote, all_bboxes)
    add_bboxes(text_blocks, BlockType.Text, all_bboxes)
    add_bboxes(title_blocks, BlockType.Title, all_bboxes)
    add_bboxes(interline_equation_blocks, BlockType.InterlineEquation, all_bboxes)

    """block嵌套问题解决"""
    """文本框与标题框重叠，优先信任文本框"""
    all_bboxes = fix_text_overlap_title_blocks(all_bboxes)
    """任何框体与舍弃框重叠，优先信任舍弃框"""
    all_bboxes = remove_need_drop_blocks(all_bboxes, discarded_blocks)

    # interline_equation 与title或text框冲突的情况，分两种情况处理
    """interline_equation框与文本类型框iou比较接近1的时候，信任行间公式框"""
    all_bboxes = fix_interline_equation_overlap_text_blocks_with_hi_iou(all_bboxes)
    """interline_equation框被包含在文本类型框内，且interline_equation比文本区块小很多时信任文本框，这时需要舍弃公式框"""
    # 通过后续大框套小框逻辑删除

    """discarded_blocks"""
    all_discarded_blocks = []
    add_bboxes(discarded_blocks, BlockType.Discarded, all_discarded_blocks)

    """footnote识别：宽度超过1/3页面宽度的，高度超过10的，处于页面下半50%区域的"""
    footnote_blocks = []
    for discarded in discarded_blocks:
        x0, y0, x1, y1 = discarded['bbox']
        if (x1 - x0) > (page_w / 3) and (y1 - y0) > 10 and y0 > (page_h / 2):
            footnote_blocks.append([x0, y0, x1, y1])

    """移除在footnote下面的任何框"""
    need_remove_blocks = find_blocks_under_footnote(all_bboxes, footnote_blocks)
    if len(need_remove_blocks) > 0:
        for block in need_remove_blocks:
            all_bboxes.remove(block)
            all_discarded_blocks.append(block)

    """经过以上处理后，还存在大框套小框的情况，则删除小框"""
    all_bboxes = remove_overlaps_min_blocks(all_bboxes)
    all_discarded_blocks = remove_overlaps_min_blocks(all_discarded_blocks)
    """将剩余的bbox做分离处理，防止后面分layout时出错"""
    # all_bboxes, drop_reasons = remove_overlap_between_bbox_for_block(all_bboxes)
    all_bboxes.sort(key=lambda x: x[0]+x[1])
    return all_bboxes, all_discarded_blocks


def find_blocks_under_footnote(all_bboxes: List[list], footnote_blocks: List[list]) -> List[list]:
    """查找位于脚注下方的blocks。

    如果一个block的纵向投影与脚注的纵向投影重叠度超过80%，且该block位于脚注下方，
    则该block将被标记为需要移除。

    Args:
        all_bboxes (list): 所有边界框的列表。
        footnote_blocks (list): 脚注边界框列表。

    Returns:
        list: 需要移除的blocks列表。
    """
    need_remove_blocks: List[list] = []
    processed_blocks: Set[tuple] = set()  # 用于跟踪已处理的blocks
    
    for block in all_bboxes:
        block_x0, block_y0, block_x1, block_y1 = block[:4]
        block_key = (block_x0, block_y0, block_x1, block_y1)  # 创建block的唯一标识
        
        if block_key in processed_blocks:
            continue
            
        for footnote_bbox in footnote_blocks:
            footnote_x0, footnote_y0, footnote_x1, footnote_y1 = footnote_bbox
            block_bbox = (block_x0, block_y0, block_x1, block_y1)
            
            # 如果footnote的纵向投影覆盖了block的纵向投影的80%且block的y0大于等于footnote的y1
            if (
                block_y0 >= footnote_y1
                and calculate_vertical_projection_overlap_ratio(block_bbox, footnote_bbox) >= 0.8
            ):
                need_remove_blocks.append(block)
                processed_blocks.add(block_key)
                break
    return need_remove_blocks

def fix_interline_equation_overlap_text_blocks_with_hi_iou(all_bboxes: List[list]) -> List[list]:
    """处理行间公式与文本块高度重叠的情况。

    当行间公式块与文本块的IOU（交并比）超过0.8时，移除文本块，保留行间公式块。

    Args:
        all_bboxes (list): 所有边界框的列表。

    Returns:
        list: 处理后的边界框列表。
    """
    # 先提取所有text和interline block
    text_blocks = []
    for block in all_bboxes:
        if block[7] == BlockType.Text:
            text_blocks.append(block)
    interline_equation_blocks = []
    for block in all_bboxes:
        if block[7] == BlockType.InterlineEquation:
            interline_equation_blocks.append(block)

    need_remove = []

    for interline_equation_block in interline_equation_blocks:
        for text_block in text_blocks:
            interline_equation_block_bbox = interline_equation_block[:4]
            text_block_bbox = text_block[:4]
            if calculate_iou(interline_equation_block_bbox, text_block_bbox) > 0.8:
                if text_block not in need_remove:
                    need_remove.append(text_block)

    if len(need_remove) > 0:
        for block in need_remove:
            all_bboxes.remove(block)

    return all_bboxes


def fix_text_overlap_title_blocks(all_bboxes: List[list]) -> List[list]:
    """处理文本块与标题块重叠的情况。

    当文本块与标题块的IOU（交并比）超过0.8时，移除标题块，保留文本块。

    Args:
        all_bboxes (list): 所有边界框的列表。

    Returns:
        list: 处理后的边界框列表。
    """
    # 先提取所有text和title block
    text_blocks = []
    for block in all_bboxes:
        if block[7] == BlockType.Text:
            text_blocks.append(block)
    title_blocks = []
    for block in all_bboxes:
        if block[7] == BlockType.Title:
            title_blocks.append(block)

    need_remove = []

    for text_block in text_blocks:
        for title_block in title_blocks:
            text_block_bbox = text_block[:4]
            title_block_bbox = title_block[:4]
            if calculate_iou(text_block_bbox, title_block_bbox) > 0.8:
                if title_block not in need_remove:
                    need_remove.append(title_block)

    if len(need_remove) > 0:
        for block in need_remove:
            all_bboxes.remove(block)

    return all_bboxes


def remove_need_drop_blocks(all_bboxes: List[list], discarded_blocks: List[dict]) -> List[list]:
    need_remove = []
    for block in all_bboxes:
        for discarded_block in discarded_blocks:
            block_bbox = block[:4]
            if (
                calculate_overlap_area_in_bbox1_area_ratio(
                    block_bbox, discarded_block['bbox']
                )
                > 0.6
            ):
                if block not in need_remove:
                    need_remove.append(block)
                    break

    if len(need_remove) > 0:
        for block in need_remove:
            all_bboxes.remove(block)
    return all_bboxes


def remove_overlaps_min_blocks(all_bboxes: List[list]) -> List[list]:
    need_remove: List[list] = []
    processed_blocks: Set[tuple] = set()  # 用于跟踪已处理的blocks
    
    for block1 in all_bboxes:
        block1_key = tuple(block1[:4])  # 创建block1的唯一标识
        if block1_key in processed_blocks:
            continue
            
        for block2 in all_bboxes:
            if block1 != block2:
                block2_key = tuple(block2[:4])  # 创建block2的唯一标识
                if block2_key in processed_blocks:
                    continue
                    
                block1_bbox = block1[:4]
                block2_bbox = block2[:4]
                overlap_box = get_minbox_if_overlap_by_ratio(
                    block1_bbox, block2_bbox, 0.8
                )
                if overlap_box is not None:
                    block_to_remove = next(
                        (block for block in all_bboxes if tuple(block[:4]) == tuple(overlap_box)),
                        None,
                    )
                    if block_to_remove is not None and block_to_remove not in need_remove:
                        large_block = block1 if block1 != block_to_remove else block2
                        x1, y1, x2, y2 = large_block[:4]
                        sx1, sy1, sx2, sy2 = block_to_remove[:4]
                        
                        # 更新边界框坐标，确保合并后的边界框完全包含两个原始block
                        large_block[:4] = [
                            min(x1, sx1),
                            min(y1, sy1),
                            max(x2, sx2),
                            max(y2, sy2)
                        ]
                        need_remove.append(block_to_remove)
                        processed_blocks.add(tuple(block_to_remove[:4]))
                        processed_blocks.add(block1_key)

    if need_remove:
        for block in need_remove:
            if block in all_bboxes:  # 确保block仍在列表中
                all_bboxes.remove(block)

    return all_bboxes
