"""胡萝卜固定测量区域面积离线验证。

该模块只做二阶段辅助分析。调用方必须先通过 YOLO 或人工确认目标为
carrot；结果不参与当前事件、库存或 App 主链路。
"""
import cv2
import numpy as np


DEFAULT_LOWER_HSV = (5, 80, 60)
DEFAULT_UPPER_HSV = (25, 255, 255)
DEFAULT_LEVEL_THRESHOLDS = (0.05, 0.30, 0.65)


def crop_roi(image_bgr, roi):
    """按固定 (x, y, w, h) 裁剪测量区域；roi=None 时使用整图。"""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError('image_bgr is empty')
    if roi is None:
        return image_bgr
    x, y, w, h = (int(v) for v in roi)
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError('roi must be a positive (x, y, w, h)')
    if x + w > image_bgr.shape[1] or y + h > image_bgr.shape[0]:
        raise ValueError('roi exceeds image bounds')
    return image_bgr[y:y + h, x:x + w]


def segment_carrot(image_bgr, roi=None, lower_hsv=DEFAULT_LOWER_HSV,
                   upper_hsv=DEFAULT_UPPER_HSV, kernel_size=5,
                   min_component_area=100):
    """返回固定测量区域内的胡萝卜橙色候选掩膜。"""
    measured = crop_roi(image_bgr, roi)
    hsv = cv2.cvtColor(measured, cv2.COLOR_BGR2HSV)
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


def measure_carrot_area(image_bgr, roi=None, **segment_kwargs):
    """计算固定测量区域内的橙色候选面积。"""
    mask = segment_carrot(image_bgr, roi=roi, **segment_kwargs)
    area = int(cv2.countNonZero(mask))
    roi_area = int(mask.shape[0] * mask.shape[1])
    return {
        'area_px': area,
        'roi_area_px': roi_area,
        'occupancy_ratio': area / roi_area if roi_area else 0.0,
        'mask': mask,
    }


def calibrated_ratio(area_px, empty_area_px=0, full_area_px=None):
    area = max(0.0, float(area_px) - float(empty_area_px or 0))
    if full_area_px is None:
        return None
    span = float(full_area_px) - float(empty_area_px or 0)
    if span <= 0:
        raise ValueError('full_area_px must be greater than empty_area_px')
    return min(1.0, max(0.0, area / span))


def quantity_level(ratio, thresholds=DEFAULT_LEVEL_THRESHOLDS):
    none_max, small_max, medium_max = thresholds
    if ratio < none_max:
        return '无'
    if ratio < small_max:
        return '少量'
    if ratio < medium_max:
        return '适量'
    return '大量'


def compare_carrot_area(before_bgr, after_bgr, roi,
                        change_threshold=0.15, empty_area_px=0,
                        full_area_px=None, **segment_kwargs):
    """比较同一固定 ROI 的动作前后胡萝卜面积和等级。"""
    if before_bgr.shape != after_bgr.shape:
        raise ValueError('before and after image shapes must match')
    before = measure_carrot_area(before_bgr, roi=roi, **segment_kwargs)
    after = measure_carrot_area(after_bgr, roi=roi, **segment_kwargs)
    delta = after['area_px'] - before['area_px']
    change_ratio = delta / max(before['area_px'], after['area_px'], 1)
    if change_ratio >= change_threshold:
        direction = 'PUT_IN'
    elif change_ratio <= -change_threshold:
        direction = 'TAKE_OUT'
    else:
        direction = 'NO_LEVEL_CHANGE'

    result = {'direction': direction, 'change_ratio': change_ratio,
              'delta_area_px': delta, 'roi': list(roi)}
    for key, measured in (('before', before), ('after', after)):
        ratio = calibrated_ratio(
            measured['area_px'], empty_area_px, full_area_px)
        level_ratio = ratio if ratio is not None else measured['occupancy_ratio']
        result[key] = {
            'area_px': measured['area_px'],
            'occupancy_ratio': measured['occupancy_ratio'],
            'calibrated_ratio': ratio,
            'level': quantity_level(level_ratio),
            'mask': measured['mask'],
        }
    return result


def mask_overlay(image_bgr, roi, mask, color=(0, 140, 255), alpha=0.45):
    """生成固定测量区域的掩膜叠加图。"""
    measured = crop_roi(image_bgr, roi).copy()
    colored = np.zeros_like(measured)
    colored[:] = color
    selected = mask > 0
    measured[selected] = cv2.addWeighted(
        measured[selected], 1.0 - alpha, colored[selected], alpha, 0)
    return measured
