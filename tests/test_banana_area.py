import cv2
import numpy as np

from banana_area import (compare_banana_area, compare_banana_region,
                         measure_banana_area, quantity_level)


def _scene(width):
    image = np.full((120, 200, 3), 40, dtype=np.uint8)
    if width:
        # BGR yellow; HSV 落在默认香蕉黄色范围内。
        cv2.rectangle(image, (20, 30), (20 + width, 90), (0, 255, 255), -1)
    return image


def test_measure_banana_area_ignores_dark_background():
    result = measure_banana_area(_scene(60))
    assert result['area_px'] > 3000
    assert result['occupancy_ratio'] < 0.3


def test_compare_detects_put_in_and_take_out():
    empty = _scene(0)
    full = _scene(120)
    put_in = compare_banana_area(empty, full)
    take_out = compare_banana_area(full, empty)
    assert put_in['direction'] == 'PUT_IN'
    assert take_out['direction'] == 'TAKE_OUT'


def test_small_area_jitter_is_ignored():
    result = compare_banana_area(_scene(100), _scene(105))
    assert result['direction'] == 'NO_LEVEL_CHANGE'


def test_region_compare_requires_banana_yolo_gate():
    before = np.full((200, 300, 3), 40, dtype=np.uint8)
    after = before.copy()
    cv2.rectangle(after, (80, 60), (220, 140), (0, 255, 255), -1)
    result = compare_banana_region(
        before, after, (50, 40, 200, 120), yolo_class_ids=[9],
        banana_class_id=2, excluded_class_ids={19}, min_delta_area_px=1000)
    assert result['direction'] == 'PUT_IN'
    assert result['is_banana_change'] is False


def test_region_compare_accepts_yellow_change_after_banana_yolo():
    before = np.full((200, 300, 3), 40, dtype=np.uint8)
    after = before.copy()
    cv2.rectangle(after, (80, 60), (220, 140), (0, 255, 255), -1)
    result = compare_banana_region(
        before, after, (50, 40, 200, 120), yolo_class_ids=[2],
        banana_class_id=2, excluded_class_ids={19}, min_delta_area_px=1000)
    assert result['direction'] == 'PUT_IN'
    assert result['is_banana_change'] is True


def test_region_compare_never_overrides_explicit_egg_event():
    before = np.full((200, 300, 3), 40, dtype=np.uint8)
    after = before.copy()
    cv2.rectangle(after, (80, 60), (220, 140), (0, 255, 255), -1)
    result = compare_banana_region(
        before, after, (0, 0, 300, 200), yolo_class_ids=[19],
        banana_class_id=2, excluded_class_ids={19}, min_delta_area_px=1000)
    assert result['direction'] == 'PUT_IN'
    assert result['is_banana_change'] is False


def test_banana_move_with_same_area_is_not_inventory_change():
    before = np.full((200, 300, 3), 40, dtype=np.uint8)
    after = before.copy()
    cv2.rectangle(before, (30, 60), (130, 140), (0, 255, 255), -1)
    cv2.rectangle(after, (160, 60), (260, 140), (0, 255, 255), -1)
    result = compare_banana_region(
        before, after, (0, 30, 300, 150), yolo_class_ids=[2],
        banana_class_id=2, excluded_class_ids={19}, min_delta_area_px=1000)
    assert result['direction'] == 'NO_LEVEL_CHANGE'
    assert result['is_banana_change'] is False


def test_calibrated_levels():
    empty_area = measure_banana_area(_scene(0))['area_px']
    full_area = measure_banana_area(_scene(120))['area_px']
    result = compare_banana_area(
        _scene(30), _scene(90),
        empty_area_px=empty_area, full_area_px=full_area)
    assert result['before']['level'] == '少量'
    assert result['after']['level'] == '大量'


def test_quantity_level_boundaries():
    assert quantity_level(0.05) == '无'
    assert quantity_level(0.20) == '少量'
    assert quantity_level(0.50) == '适量'
    assert quantity_level(0.80) == '大量'
