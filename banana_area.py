"""香蕉区域面积离线估计。

该模块只做二阶段辅助面积分析，不参与当前库存和事件主链路。
调用方应先通过 YOLO 或人工标注确认裁剪区域中的目标是 banana。
"""
import cv2
import numpy as np


DEFAULT_LOWER_HSV = (15, 70, 60)
DEFAULT_UPPER_HSV = (40, 255, 255)
DEFAULT_LEVEL_THRESHOLDS = (0.10, 0.35, 0.70)


def segment_banana(image_bgr, lower_hsv=DEFAULT_LOWER_HSV,
                   upper_hsv=DEFAULT_UPPER_HSV, kernel_size=5,
                   min_component_area=100):
    """返回香蕉黄色候选掩膜；不负责判断图中物体类别。"""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError('image_bgr is empty')

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.asarray(lower_hsv, dtype=np.uint8),
        np.asarray(upper_hsv, dtype=np.uint8))

    kernel_size = max(1, int(kernel_size))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
            cleaned[labels == label] = 255
    return cleaned


def measure_banana_area(image_bgr, **segment_kwargs):
    """计算黄色候选像素面积、占裁剪图比例及掩膜。"""
    mask = segment_banana(image_bgr, **segment_kwargs)
    area = int(cv2.countNonZero(mask))
    image_area = int(mask.shape[0] * mask.shape[1])
    return {
        'area_px': area,
        'image_area_px': image_area,
        'occupancy_ratio': area / image_area if image_area else 0.0,
        'mask': mask,
    }


def calibrated_ratio(area_px, empty_area_px=0, full_area_px=None):
    """将像素面积映射到 0~1；有满量校准时使用校准值。"""
    area = max(0.0, float(area_px) - float(empty_area_px or 0))
    if full_area_px is None:
        return None
    span = float(full_area_px) - float(empty_area_px or 0)
    if span <= 0:
        raise ValueError('full_area_px must be greater than empty_area_px')
    return min(1.0, max(0.0, area / span))


def quantity_level(ratio, thresholds=DEFAULT_LEVEL_THRESHOLDS):
    """把 0~1 占比映射为无、少量、适量、大量。"""
    none_max, small_max, medium_max = thresholds
    if ratio < none_max:
        return '无'
    if ratio < small_max:
        return '少量'
    if ratio < medium_max:
        return '适量'
    return '大量'


def compare_banana_area(before_bgr, after_bgr, change_threshold=0.15,
                        empty_area_px=0, full_area_px=None, **segment_kwargs):
    """比较动作前后香蕉面积，返回方向、面积变化和等级。"""
    before = measure_banana_area(before_bgr, **segment_kwargs)
    after = measure_banana_area(after_bgr, **segment_kwargs)
    delta = after['area_px'] - before['area_px']
    base = max(before['area_px'], after['area_px'], 1)
    change_ratio = delta / base
    if change_ratio >= change_threshold:
        direction = 'PUT_IN'
    elif change_ratio <= -change_threshold:
        direction = 'TAKE_OUT'
    else:
        direction = 'NO_LEVEL_CHANGE'

    before_calibrated = calibrated_ratio(
        before['area_px'], empty_area_px, full_area_px)
    after_calibrated = calibrated_ratio(
        after['area_px'], empty_area_px, full_area_px)
    before_level_ratio = (
        before_calibrated if before_calibrated is not None
        else before['occupancy_ratio'])
    after_level_ratio = (
        after_calibrated if after_calibrated is not None
        else after['occupancy_ratio'])

    return {
        'direction': direction,
        'change_ratio': change_ratio,
        'delta_area_px': delta,
        'before': {
            'area_px': before['area_px'],
            'occupancy_ratio': before['occupancy_ratio'],
            'calibrated_ratio': before_calibrated,
            'level': quantity_level(before_level_ratio),
            'mask': before['mask'],
        },
        'after': {
            'area_px': after['area_px'],
            'occupancy_ratio': after['occupancy_ratio'],
            'calibrated_ratio': after_calibrated,
            'level': quantity_level(after_level_ratio),
            'mask': after['mask'],
        },
    }


def mask_overlay(image_bgr, mask, color=(0, 255, 255), alpha=0.45):
    """生成用于人工检查分割范围的叠加图。"""
    overlay = image_bgr.copy()
    colored = np.zeros_like(image_bgr)
    colored[:] = color
    selected = mask > 0
    overlay[selected] = cv2.addWeighted(
        image_bgr[selected], 1.0 - alpha, colored[selected], alpha, 0)
    return overlay
