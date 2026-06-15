import cv2
import numpy as np

from carrot_area import (compare_carrot_area, measure_carrot_area,
                         quantity_level)


ROI = (20, 20, 160, 80)


def _scene(width):
    image = np.full((120, 220, 3), 35, dtype=np.uint8)
    if width:
        cv2.rectangle(image, (30, 35), (30 + width, 85),
                      (0, 140, 255), -1)
    return image


def test_fixed_roi_ignores_orange_outside_measurement_area():
    image = _scene(0)
    cv2.rectangle(image, (190, 20), (215, 90), (0, 140, 255), -1)
    result = measure_carrot_area(image, roi=ROI)
    assert result['area_px'] == 0


def test_compare_detects_put_in_take_out_and_small_jitter():
    assert compare_carrot_area(_scene(0), _scene(120), ROI)['direction'] == 'PUT_IN'
    assert compare_carrot_area(_scene(120), _scene(0), ROI)['direction'] == 'TAKE_OUT'
    assert compare_carrot_area(
        _scene(100), _scene(105), ROI)['direction'] == 'NO_LEVEL_CHANGE'


def test_calibrated_levels_use_fixed_roi_area():
    empty = measure_carrot_area(_scene(0), roi=ROI)['area_px']
    full = measure_carrot_area(_scene(140), roi=ROI)['area_px']
    result = compare_carrot_area(
        _scene(20), _scene(90), ROI,
        empty_area_px=empty, full_area_px=full)
    assert result['before']['level'] == '少量'
    assert result['after']['level'] == '适量'


def test_quantity_level_boundaries():
    assert quantity_level(0.01) == '无'
    assert quantity_level(0.10) == '少量'
    assert quantity_level(0.40) == '适量'
    assert quantity_level(0.80) == '大量'
