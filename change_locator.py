"""变化定位：差分两张静止图，定位并识别变化区域"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ChangedRegion:
    bbox: tuple                 # (x, y, w, h)
    kind: str                   # APPEAR | DISAPPEAR | REPLACE | SAME
    ref_ident: dict             # classify_fn 对 ref 裁剪图的结果，可为 None
    new_ident: dict             # classify_fn 对 new 裁剪图的结果，可为 None


def find_change_regions(ref_bgr, new_bgr, diff_thresh=30, min_area=400):
    """差分 ref/new，返回变化区域的 bbox 列表 [(x, y, w, h), ...]"""
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(ref_gray, new_gray)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 连碎块

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            regions.append((x, y, w, h))
    return regions


def classify_regions(ref_bgr, new_bgr, bboxes, classify_fn):
    """对每个变化区域裁剪并识别，判定 APPEAR/DISAPPEAR/REPLACE/SAME"""
    regions = []
    for (x, y, w, h) in bboxes:
        ref_crop = ref_bgr[y:y + h, x:x + w]
        new_crop = new_bgr[y:y + h, x:x + w]
        ref_id = classify_fn(ref_crop)
        new_id = classify_fn(new_crop)

        if ref_id is None and new_id is None:
            continue  # 噪声/阴影，丢弃
        elif ref_id is None:
            kind = 'APPEAR'
        elif new_id is None:
            kind = 'DISAPPEAR'
        elif ref_id['coarse'] != new_id['coarse']:
            kind = 'REPLACE'
        else:
            kind = 'SAME'
        regions.append(ChangedRegion((x, y, w, h), kind, ref_id, new_id))
    return regions
