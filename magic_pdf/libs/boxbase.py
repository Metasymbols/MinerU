import math


def _is_in_or_part_overlap(box1, box2) -> bool:
    """判断两个边界框(bbox)是否有部分重叠或包含关系。

    Args:
        box1: 第一个边界框，格式为 [x0, y0, x1, y1]，表示左上角和右下角坐标
        box2: 第二个边界框，格式同box1

    Returns:
        bool: 如果两个边界框有重叠或包含关系返回True，否则返回False
    """
    if box1 is None or box2 is None:
        return False

    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    return not (x1_1 < x0_2 or  # box1在box2的左边
                x0_1 > x1_2 or  # box1在box2的右边
                y1_1 < y0_2 or  # box1在box2的上边
                y0_1 > y1_2)  # box1在box2的下边


def _is_in_or_part_overlap_with_area_ratio(box1,
                                           box2,
                                           area_ratio_threshold=0.6):
    """判断两个边界框的重叠程度是否超过指定阈值。

    Args:
        box1: 第一个边界框，格式为 [x0, y0, x1, y1]，表示左上角和右下角坐标
        box2: 第二个边界框，格式同box1
        area_ratio_threshold: 重叠面积比例阈值，默认为0.6

    Returns:
        bool: 如果box1和box2的重叠面积占box1面积的比例超过阈值，返回True，否则返回False
    """
    if box1 is None or box2 is None:
        return False

    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    if not _is_in_or_part_overlap(box1, box2):
        return False

    # 计算重叠面积
    x_left = max(x0_1, x0_2)
    y_top = max(y0_1, y0_2)
    x_right = min(x1_1, x1_2)
    y_bottom = min(y1_1, y1_2)
    overlap_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算box1的面积
    box1_area = (x1_1 - x0_1) * (y1_1 - y0_1)

    return overlap_area / box1_area > area_ratio_threshold


def _is_in(box1, box2) -> bool:
    """判断一个边界框是否完全包含在另一个边界框内。

    Args:
        box1: 待判断的边界框，格式为 [x0, y0, x1, y1]，表示左上角和右下角坐标
        box2: 参考边界框，格式同box1

    Returns:
        bool: 如果box1完全在box2内部返回True，否则返回False
    """
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    return (x0_1 >= x0_2 and  # box1的左边界不在box2的左边外
            y0_1 >= y0_2 and  # box1的上边界不在box2的上边外
            x1_1 <= x1_2 and  # box1的右边界不在box2的右边外
            y1_1 <= y1_2)  # box1的下边界不在box2的下边外


def _is_part_overlap(box1, box2) -> bool:
    """判断两个边界框是否部分重叠但不完全包含。

    Args:
        box1: 第一个边界框，格式为 [x0, y0, x1, y1]，表示左上角和右下角坐标
        box2: 第二个边界框，格式同box1

    Returns:
        bool: 如果两个边界框部分重叠但不完全包含返回True，否则返回False
    """
    if box1 is None or box2 is None:
        return False

    return _is_in_or_part_overlap(box1, box2) and not _is_in(box1, box2)


def _left_intersect(left_box, right_box):
    """检查两个边界框在左边界处是否相交。

    Args:
        left_box: 左侧边界框，格式为 [x0, y0, x1, y1]，表示左上角和右下角坐标
        right_box: 右侧边界框，格式同left_box

    Returns:
        bool: 如果left_box的右边界与right_box的左边界相交返回True，否则返回False
    """
    if left_box is None or right_box is None:
        return False

    x0_1, y0_1, x1_1, y1_1 = left_box
    x0_2, y0_2, x1_2, y1_2 = right_box

    return x1_1 > x0_2 and x0_1 < x0_2 and (y0_1 <= y0_2 <= y1_1
                                            or y0_1 <= y1_2 <= y1_1)


def _right_intersect(left_box, right_box):
    """检查两个边界框在右边界处是否相交。

    Args:
        left_box: 左侧边界框，格式为 [x0, y0, x1, y1]，表示左上角和右下角坐标
        right_box: 右侧边界框，格式同left_box

    Returns:
        bool: 如果left_box的左边界与right_box的右边界相交返回True，否则返回False
    """
    if left_box is None or right_box is None:
        return False

    x0_1, y0_1, x1_1, y1_1 = left_box
    x0_2, y0_2, x1_2, y1_2 = right_box

    return x0_1 < x1_2 and x1_1 > x1_2 and (y0_1 <= y0_2 <= y1_1
                                            or y0_1 <= y1_2 <= y1_1)


def _is_vertical_full_overlap(box1, box2, x_torlence=2):
    """判断两个边界框在垂直方向上是否完全重叠。

    Args:
        box1: 第一个边界框，格式为 [x0, y0, x1, y1]，表示左上角和右下角坐标
        box2: 第二个边界框，格式同box1
        x_torlence: x方向上的容差值，默认为2

    Returns:
        bool: 如果两个边界框在x方向上互相包含且在y方向上有重叠返回True，否则返回False

    Note:
        x方向上的判断条件：box1包含box2或box2包含box1，不允许部分包含
        y方向上的判断条件：box1和box2必须有重叠部分
    """
    # 解析box的坐标
    x11, y11, x12, y12 = box1  # 左上角和右下角的坐标 (x1, y1, x2, y2)
    x21, y21, x22, y22 = box2

    # 在x轴方向上，box1是否包含box2 或 box2包含box1
    contains_in_x = (x11 - x_torlence <= x21 and x12 + x_torlence >= x22) or (
        x21 - x_torlence <= x11 and x22 + x_torlence >= x12)

    # 在y轴方向上，box1和box2是否有重叠
    overlap_in_y = not (y12 < y21 or y11 > y22)

    return contains_in_x and overlap_in_y


def _is_bottom_full_overlap(box1, box2, y_tolerance=2):
    """检查两个边界框在垂直方向上是否有轻微重叠。

    Args:
        box1: 上方边界框，格式为 [x0, y0, x1, y1]，表示左上角和右下角坐标
        box2: 下方边界框，格式同box1
        y_tolerance: y方向上允许的重叠容差值，默认为2

    Returns:
        bool: 如果box1的下边界与box2的上边界有轻微重叠且x方向上基本对齐返回True，否则返回False

    Note:
        与_is_vertical_full_overlap的区别是本函数允许x方向上有轻微的重叠，更适合检测垂直排列的文本块
    """
    if box1 is None or box2 is None:
        return False

    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    tolerance_margin = 2
    is_xdir_full_overlap = (
        (x0_1 - tolerance_margin <= x0_2 <= x1_1 + tolerance_margin
         and x0_1 - tolerance_margin <= x1_2 <= x1_1 + tolerance_margin)
        or (x0_2 - tolerance_margin <= x0_1 <= x1_2 + tolerance_margin
            and x0_2 - tolerance_margin <= x1_1 <= x1_2 + tolerance_margin))

    return y0_2 < y1_1 and 0 < (y1_1 -
                                y0_2) < y_tolerance and is_xdir_full_overlap


def _is_left_overlap(
    box1,
    box2,
):
    """检查两个边界框在水平方向上是否有左侧重叠。

    Args:
        box1: 第一个边界框，格式为 [x0, y0, x1, y1]，表示左上角和右下角坐标
        box2: 第二个边界框，格式同box1

    Returns:
        bool: 如果box1的左侧与box2有重叠返回True，否则返回False

    Note:
        1. Y方向上允许部分重叠或完全重叠
        2. 不考虑box1和box2的上下位置关系
        3. 重叠判定基于投影重叠面积比例
    """

    def __overlap_y(Ay1, Ay2, By1, By2):
        return max(0, min(Ay2, By2) - max(Ay1, By1))

    if box1 is None or box2 is None:
        return False

    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    y_overlap_len = __overlap_y(y0_1, y1_1, y0_2, y1_2)
    ratio_1 = 1.0 * y_overlap_len / (y1_1 - y0_1) if y1_1 - y0_1 != 0 else 0
    ratio_2 = 1.0 * y_overlap_len / (y1_2 - y0_2) if y1_2 - y0_2 != 0 else 0
    vertical_overlap_cond = ratio_1 >= 0.5 or ratio_2 >= 0.5

    # vertical_overlap_cond = y0_1<=y0_2<=y1_1 or y0_1<=y1_2<=y1_1 or y0_2<=y0_1<=y1_2 or y0_2<=y1_1<=y1_2
    return x0_1 <= x0_2 <= x1_1 and vertical_overlap_cond


def __is_overlaps_y_exceeds_threshold(bbox1,
                                      bbox2,
                                      overlap_ratio_threshold=0.8):
    """检查两个边界框在y轴方向上的重叠程度是否超过阈值。

    Args:
        bbox1: 第一个边界框，格式为 [x0, y0, x1, y1]
        bbox2: 第二个边界框，格式同bbox1
        overlap_ratio_threshold: 重叠比例阈值，默认为0.8

    Returns:
        bool: 如果重叠区域高度占较小边界框高度的比例超过阈值，返回True，否则返回False
    """
    _, y0_1, _, y1_1 = bbox1
    _, y0_2, _, y1_2 = bbox2

    overlap = max(0, min(y1_1, y1_2) - max(y0_1, y0_2))
    height1, height2 = y1_1 - y0_1, y1_2 - y0_2
    # max_height = max(height1, height2)
    min_height = min(height1, height2)

    return (overlap / min_height) > overlap_ratio_threshold


def calculate_iou(bbox1, bbox2):
    """计算两个边界框的交并比(IOU)。

    Args:
        bbox1 (list[float]): 第一个边界框的坐标，格式为 [x1, y1, x2, y2]，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。
        bbox2 (list[float]): 第二个边界框的坐标，格式与 `bbox1` 相同。

    Returns:
        float: 两个边界框的交并比(IOU)，取值范围为 [0, 1]。
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # The area of both rectangles
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    if any([bbox1_area == 0, bbox2_area == 0]):
        return 0

    # Compute the intersection over union by taking the intersection area
    # and dividing it by the sum of both areas minus the intersection area
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

    return iou


def calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2):
    """计算两个边界框的重叠面积占较小边界框面积的比例。

    Args:
        bbox1: 第一个边界框，格式为 [x0, y0, x1, y1]
        bbox2: 第二个边界框，格式同bbox1

    Returns:
        float: 重叠面积与较小边界框面积的比值，范围[0, 1]
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    min_box_area = min([(bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]),
                        (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])])
    if min_box_area == 0:
        return 0
    else:
        return intersection_area / min_box_area


def calculate_overlap_area_in_bbox1_area_ratio(bbox1, bbox2):
    """计算box1和box2的重叠面积占bbox1的比例。

    Args:
        bbox1: 第一个边界框，格式为 [x0, y0, x1, y1]
        bbox2: 第二个边界框，格式同bbox1

    Returns:
        float: 重叠面积占bbox1面积的比例，范围[0, 1]
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    if bbox1_area == 0:
        return 0
    else:
        return intersection_area / bbox1_area


def get_minbox_if_overlap_by_ratio(bbox1, bbox2, ratio):
    """根据重叠比例返回较小的边界框。

    Args:
        bbox1: 第一个边界框，格式为 [x0, y0, x1, y1]
        bbox2: 第二个边界框，格式同bbox1
        ratio: 重叠面积比例阈值

    Returns:
        list: 如果重叠面积占较小边界框面积的比例大于ratio，返回面积较小的边界框；否则返回None
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    overlap_ratio = calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2)
    if overlap_ratio > ratio:
        if area1 <= area2:
            return bbox1
        else:
            return bbox2
    else:
        return None


def get_bbox_in_boundary(bboxes: list, boundary: tuple) -> list:
    """获取完全在指定边界内的所有边界框。

    Args:
        bboxes: 边界框列表，每个边界框格式为 [x0, y0, x1, y1]
        boundary: 边界范围，格式为 (x0, y0, x1, y1)

    Returns:
        list: 完全在boundary内的边界框列表
    """
    x0, y0, x1, y1 = boundary
    new_boxes = [
        box for box in bboxes
        if box[0] >= x0 and box[1] >= y0 and box[2] <= x1 and box[3] <= y1
    ]
    return new_boxes


def is_vbox_on_side(bbox, width, height, side_threshold=0.2):
    """判断边界框是否位于页面边缘。

    Args:
        bbox: 边界框，格式为 [x0, y0, x1, y1]
        width: 页面宽度
        height: 页面高度
        side_threshold: 边缘阈值，默认为0.2，表示页面宽度的20%

    Returns:
        bool: 如果边界框位于页面左边缘或右边缘返回True，否则返回False
    """
    x0, x1 = bbox[0], bbox[2]
    if x1 <= width * side_threshold or x0 >= width * (1 - side_threshold):
        return True
    return False


def find_top_nearest_text_bbox(pymu_blocks, obj_bbox):
    tolerance_margin = 4
    top_boxes = [
        box for box in pymu_blocks
        if obj_bbox[1] - box['bbox'][3] >= -tolerance_margin
        and not _is_in(box['bbox'], obj_bbox)
    ]
    # 然后找到X方向上有互相重叠的
    top_boxes = [
        box for box in top_boxes if any([
            obj_bbox[0] - tolerance_margin <= box['bbox'][0] <= obj_bbox[2] +
            tolerance_margin, obj_bbox[0] -
            tolerance_margin <= box['bbox'][2] <= obj_bbox[2] +
            tolerance_margin, box['bbox'][0] -
            tolerance_margin <= obj_bbox[0] <= box['bbox'][2] +
            tolerance_margin, box['bbox'][0] -
            tolerance_margin <= obj_bbox[2] <= box['bbox'][2] +
            tolerance_margin
        ])
    ]

    # 然后找到y1最大的那个
    if len(top_boxes) > 0:
        top_boxes.sort(key=lambda x: x['bbox'][3], reverse=True)
        return top_boxes[0]
    else:
        return None


def find_bottom_nearest_text_bbox(pymu_blocks, obj_bbox):
    bottom_boxes = [
        box for box in pymu_blocks if box['bbox'][1] -
        obj_bbox[3] >= -2 and not _is_in(box['bbox'], obj_bbox)
    ]
    # 然后找到X方向上有互相重叠的
    bottom_boxes = [
        box for box in bottom_boxes if any([
            obj_bbox[0] - 2 <= box['bbox'][0] <= obj_bbox[2] + 2, obj_bbox[0] -
            2 <= box['bbox'][2] <= obj_bbox[2] + 2, box['bbox'][0] -
            2 <= obj_bbox[0] <= box['bbox'][2] + 2, box['bbox'][0] -
            2 <= obj_bbox[2] <= box['bbox'][2] + 2
        ])
    ]

    # 然后找到y0最小的那个
    if len(bottom_boxes) > 0:
        bottom_boxes.sort(key=lambda x: x['bbox'][1], reverse=False)
        return bottom_boxes[0]
    else:
        return None


def find_left_nearest_text_bbox(pymu_blocks, obj_bbox):
    """寻找左侧最近的文本block."""
    left_boxes = [
        box for box in pymu_blocks if obj_bbox[0] -
        box['bbox'][2] >= -2 and not _is_in(box['bbox'], obj_bbox)
    ]
    # 然后找到X方向上有互相重叠的
    left_boxes = [
        box for box in left_boxes if any([
            obj_bbox[1] - 2 <= box['bbox'][1] <= obj_bbox[3] + 2, obj_bbox[1] -
            2 <= box['bbox'][3] <= obj_bbox[3] + 2, box['bbox'][1] -
            2 <= obj_bbox[1] <= box['bbox'][3] + 2, box['bbox'][1] -
            2 <= obj_bbox[3] <= box['bbox'][3] + 2
        ])
    ]

    # 然后找到x1最大的那个
    if len(left_boxes) > 0:
        left_boxes.sort(key=lambda x: x['bbox'][2], reverse=True)
        return left_boxes[0]
    else:
        return None


def find_right_nearest_text_bbox(pymu_blocks, obj_bbox):
    """寻找右侧最近的文本block."""
    right_boxes = [
        box for box in pymu_blocks if box['bbox'][0] -
        obj_bbox[2] >= -2 and not _is_in(box['bbox'], obj_bbox)
    ]
    # 然后找到X方向上有互相重叠的
    right_boxes = [
        box for box in right_boxes if any([
            obj_bbox[1] - 2 <= box['bbox'][1] <= obj_bbox[3] + 2, obj_bbox[1] -
            2 <= box['bbox'][3] <= obj_bbox[3] + 2, box['bbox'][1] -
            2 <= obj_bbox[1] <= box['bbox'][3] + 2, box['bbox'][1] -
            2 <= obj_bbox[3] <= box['bbox'][3] + 2
        ])
    ]

    # 然后找到x0最小的那个
    if len(right_boxes) > 0:
        right_boxes.sort(key=lambda x: x['bbox'][0], reverse=False)
        return right_boxes[0]
    else:
        return None


def bbox_relative_pos(bbox1, bbox2):
    """判断两个矩形框的相对位置关系.

    Args:
        bbox1: 一个四元组，表示第一个矩形框的左上角和右下角的坐标，格式为(x1, y1, x1b, y1b)
        bbox2: 一个四元组，表示第二个矩形框的左上角和右下角的坐标，格式为(x2, y2, x2b, y2b)

    Returns:
        一个四元组，表示矩形框1相对于矩形框2的位置关系，格式为(left, right, bottom, top)
        其中，left表示矩形框1是否在矩形框2的左侧，right表示矩形框1是否在矩形框2的右侧，
        bottom表示矩形框1是否在矩形框2的下方，top表示矩形框1是否在矩形框2的上方
    """
    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    return left, right, bottom, top


def bbox_distance(bbox1, bbox2):
    """计算两个矩形框的距离。

    Args:
        bbox1 (tuple): 第一个矩形框的坐标，格式为 (x1, y1, x2, y2)，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。
        bbox2 (tuple): 第二个矩形框的坐标，格式为 (x1, y1, x2, y2)，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。

    Returns:
        float: 矩形框之间的距离。
    """

    def dist(point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 +
                         (point1[1] - point2[1])**2)

    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    left, right, bottom, top = bbox_relative_pos(bbox1, bbox2)

    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    return 0.0


def box_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def get_overlap_area(bbox1, bbox2):
    """计算box1和box2的重叠面积占bbox1的比例."""
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    return (x_right - x_left) * (y_bottom - y_top)


def calculate_vertical_projection_overlap_ratio(block1, block2):
    """
    Calculate the proportion of the x-axis covered by the vertical projection of two blocks.

    Args:
        block1 (tuple): Coordinates of the first block (x0, y0, x1, y1).
        block2 (tuple): Coordinates of the second block (x0, y0, x1, y1).

    Returns:
        float: The proportion of the x-axis covered by the vertical projection of the two blocks.
    """
    x0_1, _, x1_1, _ = block1
    x0_2, _, x1_2, _ = block2

    # Calculate the intersection of the x-coordinates
    x_left = max(x0_1, x0_2)
    x_right = min(x1_1, x1_2)

    if x_right < x_left:
        return 0.0

    # Length of the intersection
    intersection_length = x_right - x_left

    # Length of the x-axis projection of the first block
    block1_length = x1_1 - x0_1

    if block1_length == 0:
        return 0.0

    # Proportion of the x-axis covered by the intersection
    # logger.info(f"intersection_length: {intersection_length}, block1_length: {block1_length}")
    return intersection_length / block1_length
