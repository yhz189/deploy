"""变化定位：差分两张静止图，定位并识别变化区域"""
from dataclasses import dataclass

import cv2
import numpy as np


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
