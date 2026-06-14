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

def _expand_bbox(bbox, shape, pad_ratio=0.12, min_pad=12):
    x, y, w, h = bbox
    img_h, img_w = shape[:2]
    pad = max(min_pad, int(round(max(w, h) * pad_ratio)))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)
    return x1, y1, x2 - x1, y2 - y1


def _intersects_or_close(a, b, gap=12):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + gap < bx
        or bx + bw + gap < ax
        or ay + ah + gap < by
        or by + bh + gap < ay
    )


def _merge_bboxes(bboxes):
    merged = []
    for bbox in bboxes:
        cur = bbox
        changed = True
        while changed:
            changed = False
            keep = []
            for other in merged:
                if _intersects_or_close(cur, other):
                    ox, oy, ow, oh = other
                    x1 = min(cur[0], ox)
                    y1 = min(cur[1], oy)
                    x2 = max(cur[0] + cur[2], ox + ow)
                    y2 = max(cur[1] + cur[3], oy + oh)
                    cur = (x1, y1, x2 - x1, y2 - y1)
                    changed = True
                else:
                    keep.append(other)
            merged = keep
        merged.append(cur)
    return merged


def find_change_regions(ref_bgr, new_bgr, diff_thresh=35, min_area=700):
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
            regions.append(_expand_bbox((x, y, w, h), ref_bgr.shape))
    return _merge_bboxes(regions)


def classify_regions(ref_bgr, new_bgr, bboxes, classify_fn):
    """对每个变化区域裁剪并识别，判定 APPEAR/DISAPPEAR/REPLACE/SAME"""
    regions = []
    for (x, y, w, h) in bboxes:
        ref_crop = ref_bgr[y:y + h, x:x + w]
        new_crop = new_bgr[y:y + h, x:x + w]
        ref_id = classify_fn(ref_crop)
        new_id = classify_fn(new_crop)

        if ref_id is None and new_id is None:
            regions.append(ChangedRegion((x, y, w, h), 'NOISE', None, None))
            continue
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
