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


from change_locator import classify_regions, ChangedRegion


def _fake_classify(known):
    """known: dict，键为 (x, y, w, h)，值为 ident 或 None"""
    def fn(crop):
        # 测试里用 crop 的左上角像素值编码区域 id
        key = int(crop[0, 0, 0])
        return known.get(key)
    return fn


def test_classify_appear():
    ref = np.zeros((100, 100, 3), dtype=np.uint8)
    new = np.zeros((100, 100, 3), dtype=np.uint8)
    new[0:50, 0:50, 0] = 7  # 该区域 new 有物体，ref 没有
    ident = {'class_id': 0, 'fine': 'Apple', 'coarse': '蔬果',
             'conf': 0.8, 'area': 1000.0}
    classify_fn = _fake_classify({0: None, 7: ident})
    regions = classify_regions(ref, new, [(0, 0, 50, 50)], classify_fn)
    assert len(regions) == 1
    assert regions[0].kind == 'APPEAR'
    assert regions[0].new_ident == ident


def test_classify_disappear():
    ref = np.zeros((100, 100, 3), dtype=np.uint8)
    ref[0:50, 0:50, 0] = 7
    new = np.zeros((100, 100, 3), dtype=np.uint8)
    ident = {'class_id': 0, 'fine': 'Apple', 'coarse': '蔬果',
             'conf': 0.8, 'area': 1000.0}
    classify_fn = _fake_classify({7: ident, 0: None})
    regions = classify_regions(ref, new, [(0, 0, 50, 50)], classify_fn)
    assert regions[0].kind == 'DISAPPEAR'


def test_classify_noise_returns_noise_region():
    ref = np.zeros((100, 100, 3), dtype=np.uint8)
    new = np.zeros((100, 100, 3), dtype=np.uint8)
    classify_fn = _fake_classify({0: None})  # 两侧都识别不出 → 返回 NOISE，由 event_detector 用记忆判断
    regions = classify_regions(ref, new, [(0, 0, 50, 50)], classify_fn)
    assert len(regions) == 1
    assert regions[0].kind == 'NOISE'
    assert regions[0].ref_ident is None
    assert regions[0].new_ident is None
