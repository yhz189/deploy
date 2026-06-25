"""复杂光照与反光条件下的轻量 OpenCV 离线增强工具。"""
import cv2
import numpy as np


LOW_LIGHT_THRESHOLD = 85.0
HIGHLIGHT_V_THRESHOLD = 235
HIGHLIGHT_S_THRESHOLD = 45


def image_metrics(image_bgr):
    """返回便于比较增强前后效果的基础灰度指标。"""
    _validate_image(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    p05, p95 = np.percentile(gray, [5, 95])
    return {
        'mean_brightness': round(float(gray.mean()), 3),
        'contrast_std': round(float(gray.std()), 3),
        'dynamic_range_p95_p05': round(float(p95 - p05), 3),
    }


def enhance_low_light(image_bgr, brightness_threshold=LOW_LIGHT_THRESHOLD):
    """暗图执行自动 Gamma、轻度降噪和 CLAHE；正常图保持不变。"""
    _validate_image(image_bgr)
    before = image_metrics(image_bgr)
    if before['mean_brightness'] >= brightness_threshold:
        return image_bgr.copy(), {
            'applied': False,
            'gamma': 1.0,
            'threshold': float(brightness_threshold),
            'before': before,
            'after': before,
        }

    mean = max(before['mean_brightness'], 1.0)
    target_brightness = 80.0
    gamma = float(np.clip(
        np.log(target_brightness / 255.0) / np.log(mean / 255.0),
        0.60, 0.95))
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)],
        dtype=np.uint8)
    denoised = cv2.bilateralFilter(image_bgr, 7, 35, 35)
    corrected = cv2.LUT(denoised, table)

    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(
        cv2.merge((clahe.apply(l_channel), a_channel, b_channel)),
        cv2.COLOR_LAB2BGR)
    return enhanced, {
        'applied': True,
        'gamma': round(gamma, 4),
        'threshold': float(brightness_threshold),
        'before': before,
        'after': image_metrics(enhanced),
    }


def detect_highlights(image_bgr, v_threshold=HIGHLIGHT_V_THRESHOLD,
                      s_threshold=HIGHLIGHT_S_THRESHOLD):
    """使用 HSV 高亮度、低饱和度条件检测疑似白色高光。"""
    _validate_image(image_bgr)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    relative_v_threshold = max(140, int(np.percentile(v_channel, 97)))
    effective_v_threshold = min(v_threshold, relative_v_threshold)
    effective_s_threshold = max(s_threshold, 80)
    brightness_mask = cv2.inRange(
        hsv, np.array((0, 0, effective_v_threshold), dtype=np.uint8),
        np.array((179, effective_s_threshold, 255), dtype=np.uint8))
    local_mean = cv2.GaussianBlur(v_channel, (0, 0), 9)
    local_peak = cv2.subtract(v_channel, local_mean)
    _, local_peak_mask = cv2.threshold(
        local_peak, 12, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(brightness_mask, local_peak_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = int(cv2.countNonZero(mask))
    return mask, {
        'area_px': area,
        'area_ratio': round(area / float(mask.size), 6),
        'fragment_count': len(contours),
        'v_threshold': int(effective_v_threshold),
        's_threshold': int(effective_s_threshold),
    }


def suppress_highlights(image_bgr, highlight_mask, strength=0.72):
    """压低疑似高光区域亮度，不尝试恢复已丢失的真实纹理。"""
    _validate_image(image_bgr)
    if highlight_mask.shape != image_bgr.shape[:2]:
        raise ValueError('highlight mask shape must match image')
    strength = float(np.clip(strength, 0.0, 1.0))
    expanded = cv2.dilate(highlight_mask, np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(expanded, (0, 0), 3).astype(np.float32) / 255.0
    alpha = alpha[:, :, None] * strength
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    compressed = hsv.copy()
    compressed[:, :, 2] = np.clip(
        hsv[:, :, 2].astype(np.float32) * 0.45,
        0, 255).astype(np.uint8)
    compressed_bgr = cv2.cvtColor(compressed, cv2.COLOR_HSV2BGR)
    return np.clip(
        image_bgr.astype(np.float32) * (1.0 - alpha)
        + compressed_bgr.astype(np.float32) * alpha,
        0, 255).astype(np.uint8)


def reflective_cover_risk(image_bgr, highlight_mask, highlight_stats=None):
    """根据启发式特征输出疑似反光覆盖风险，而非确定性判断。"""
    _validate_image(image_bgr)
    if highlight_mask.shape != image_bgr.shape[:2]:
        raise ValueError('highlight mask shape must match image')
    if highlight_stats is None:
        _, highlight_stats = detect_highlights(image_bgr)

    edges = cv2.Canny(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), 80, 160)
    dilated_mask = cv2.dilate(highlight_mask, np.ones((3, 3), np.uint8))
    covered = max(cv2.countNonZero(dilated_mask), 1)
    edge_density = cv2.countNonZero(cv2.bitwise_and(edges, dilated_mask)) / covered
    area_score = min(float(highlight_stats['area_ratio']) / 0.12, 1.0)
    fragment_score = min(float(highlight_stats['fragment_count']) / 12.0, 1.0)
    edge_score = min(float(edge_density) / 0.18, 1.0)
    score = float(np.clip(
        0.50 * area_score + 0.25 * fragment_score + 0.25 * edge_score,
        0.0, 1.0))
    return {
        'possible_reflective_cover': score >= 0.45,
        'risk_score': round(score, 4),
        'edge_density_near_highlights': round(float(edge_density), 6),
    }


def analyze_complex_conditions(image_bgr):
    """运行完整离线分析，返回生成图和可序列化指标。"""
    enhanced, low_light = enhance_low_light(image_bgr)
    mask, highlights = detect_highlights(image_bgr)
    suppressed = suppress_highlights(image_bgr, mask)
    source_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    suppressed_gray = cv2.cvtColor(suppressed, cv2.COLOR_BGR2GRAY)
    selected = mask > 0
    if np.any(selected):
        before_mean = float(source_gray[selected].mean())
        after_mean = float(suppressed_gray[selected].mean())
        reduction = (before_mean - after_mean) / max(before_mean, 1.0)
    else:
        before_mean = after_mean = reduction = 0.0
    highlights['suppression'] = {
        'mean_brightness_before': round(before_mean, 3),
        'mean_brightness_after': round(after_mean, 3),
        'brightness_reduction_ratio': round(float(reduction), 4),
    }
    risk = reflective_cover_risk(image_bgr, mask, highlights)
    result = {
        'mode': 'offline_demonstration',
        'source_metrics': image_metrics(image_bgr),
        'low_light': low_light,
        'highlights': highlights,
        'reflective_cover_risk': risk,
        'limitations': [
            'highlight_suppressed only reduces glare brightness and does not restore lost texture',
            'possible_reflective_cover is a heuristic risk signal, not proof of plastic film',
            'physical polarization requires a polarizing filter',
        ],
    }
    images = {
        'low_light_enhanced': enhanced,
        'highlight_mask': mask,
        'highlight_suppressed': suppressed,
    }
    return images, result


def _gamma_transform(image_bgr, gamma):
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)],
        dtype=np.uint8)
    return cv2.LUT(image_bgr, table)


def _validate_image(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError('image must not be empty')
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError('image must be a BGR color image')
