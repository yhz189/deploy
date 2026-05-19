import numpy as np
from change_locator import find_change_regions


def test_no_change_returns_empty():
    shelf = np.full((480, 640, 3), 100, dtype=np.uint8)
    assert find_change_regions(shelf, shelf) == []


def test_one_added_object_returns_one_region():
    ref = np.full((480, 640, 3), 100, dtype=np.uint8)
    new = ref.copy()
    new[200:300, 250:350] = 230  # 加一个 100x100 的色块
    regions = find_change_regions(ref, new)
    assert len(regions) == 1
    x, y, w, h = regions[0]
    # 变化区域应大致覆盖色块位置
    assert 230 <= x <= 270
    assert 190 <= y <= 270
    assert w >= 70 and h >= 70


def test_tiny_change_filtered_by_min_area():
    ref = np.full((480, 640, 3), 100, dtype=np.uint8)
    new = ref.copy()
    new[10:18, 10:18] = 230  # 8x8 小块，低于 min_area
    assert find_change_regions(ref, new) == []
