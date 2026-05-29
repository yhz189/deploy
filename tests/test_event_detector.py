import numpy as np
from event_detector import EventDetector

STILL = np.zeros((8, 8, 3), dtype=np.uint8)
MOVING = np.zeros((8, 8, 3), dtype=np.uint8)
MOVING[0, 0, 0] = 255  # 用首像素编码「该帧处于运动中」


def _fake_motion(prev, cur):
    return bool(cur[0, 0, 0])


def _fake_locate_empty(ref, new):
    return []


def _feed(det, frame, n):
    last = (None, det.state)
    for _ in range(n):
        last = det.update(frame)
    return last


def test_stable_to_busy_after_enter_frames():
    det = EventDetector(_fake_motion, _fake_locate_empty,
                        enter_frames=3, exit_frames=10, settle_frames=5)
    det.seed(STILL, [])
    _feed(det, STILL, 2)
    assert det.state == 'STABLE'
    _feed(det, MOVING, 3)
    assert det.state == 'BUSY'


def test_full_cycle_back_to_stable():
    det = EventDetector(_fake_motion, _fake_locate_empty,
                        enter_frames=3, exit_frames=10, settle_frames=5)
    det.seed(STILL, [])
    _feed(det, MOVING, 3)        # → BUSY
    _feed(det, STILL, 10)        # → SETTLING
    events, state = _feed(det, STILL, 5)  # SETTLING 确认 → 分析 → STABLE
    assert state == 'STABLE'
    assert events == []          # 空 locate → 无事件


def test_motion_during_settling_returns_to_busy():
    det = EventDetector(_fake_motion, _fake_locate_empty,
                        enter_frames=3, exit_frames=10, settle_frames=5)
    det.seed(STILL, [])
    _feed(det, MOVING, 3)
    _feed(det, STILL, 10)        # → SETTLING
    _feed(det, MOVING, 1)        # SETTLING 期间又动 → 回 BUSY
    assert det.state == 'BUSY'


from change_locator import ChangedRegion


def _ident(class_id, fine, coarse, area):
    return {'class_id': class_id, 'fine': fine, 'coarse': coarse,
            'conf': 0.8, 'area': area}


def _run_analyze(regions):
    """构造一个 detector，直接驱动一次分析，返回事件列表"""
    det = EventDetector(_fake_motion, lambda r, n: regions)
    det.seed(STILL, [])
    return det._analyze(STILL), det


def test_appear_emits_put_in():
    apple = _ident(0, 'Apple', '蔬果', 1000.0)
    region = ChangedRegion((10, 10, 40, 40), 'APPEAR', None, apple)
    events, det = _run_analyze([region])
    assert events == [('PUT_IN', {'added': {0: 1}})]
    assert len(det.placed_items) == 1


def test_take_out_uses_memory():
    apple = _ident(0, 'Apple', '蔬果', 1000.0)
    det = EventDetector(_fake_motion, None)
    det.seed(STILL, [{'class_id': 0, 'fine': 'Apple',
                      'coarse': '蔬果', 'bbox': (10, 10, 40, 40)}])
    region = ChangedRegion((12, 12, 40, 40), 'DISAPPEAR', apple, None)
    events = det._analyze_regions([region])
    assert events == [('TAKE_OUT', {'removed': {0: 1}})]
    assert det.placed_items == []


def test_rearrange_is_suppressed():
    # 同粗类、尺寸相近的一出一进 → 判整理，不计事件
    apple_a = _ident(0, 'Apple', '蔬果', 1000.0)
    apple_b = _ident(0, 'Apple', '蔬果', 1000.0)
    det = EventDetector(_fake_motion, None)
    det.seed(STILL, [{'class_id': 0, 'fine': 'Apple',
                      'coarse': '蔬果', 'bbox': (10, 10, 40, 40)}])
    regions = [
        ChangedRegion((10, 10, 40, 40), 'DISAPPEAR', apple_a, None),
        ChangedRegion((200, 200, 40, 40), 'APPEAR', None, apple_b),
    ]
    events = det._analyze_regions(regions)
    assert events == []
    assert len(det.placed_items) == 1  # 物品仍在，位置已更新


def test_partial_take_out_on_area_shrink():
    # 同位置同粗类、检测框面积明显变小 → 部分取出
    big = _ident(0, 'Apple', '蔬果', 1000.0)
    small = _ident(0, 'Apple', '蔬果', 500.0)
    det = EventDetector(_fake_motion, None)
    det.seed(STILL, [{'class_id': 0, 'fine': 'Apple',
                      'coarse': '蔬果', 'bbox': (10, 10, 40, 40)}])
    region = ChangedRegion((10, 10, 40, 40), 'SAME', big, small)
    events = det._analyze_regions([region])
    assert events == [('PARTIAL_TAKE_OUT', {'removed': {0: 1}})]


def test_take_out_falls_back_to_ref_ident_when_memory_misses():
    # 记忆为空（未命中），但 ref_ident 已识别出旧物品 → 仍应发出 TAKE_OUT
    apple = _ident(0, 'Apple', '蔬果', 1000.0)
    det = EventDetector(_fake_motion, None)
    det.seed(STILL, [])  # 空记忆
    region = ChangedRegion((10, 10, 40, 40), 'DISAPPEAR', apple, None)
    events = det._analyze_regions([region])
    assert events == [('TAKE_OUT', {'removed': {0: 1}})]


def test_take_out_uses_ref_ident_count():
    egg = _ident(19, 'egg', '肉蛋生鲜', 1000.0)
    egg['count'] = 3
    det = EventDetector(_fake_motion, None)
    det.seed(STILL, [
        {'class_id': 19, 'fine': 'egg', 'coarse': '肉蛋生鲜',
         'bbox': (10, 10, 40, 40)}
        for _ in range(3)
    ])
    region = ChangedRegion((10, 10, 40, 40), 'DISAPPEAR', egg, None)
    events = det._analyze_regions([region])
    assert events == [('TAKE_OUT', {'removed': {19: 3}})]
    assert det.placed_items == []


def test_same_region_uses_count_delta():
    before = _ident(19, 'egg', '肉蛋生鲜', 1000.0)
    before['count'] = 5
    after = _ident(19, 'egg', '肉蛋生鲜', 1000.0)
    after['count'] = 3
    det = EventDetector(_fake_motion, None)
    det.seed(STILL, [
        {'class_id': 19, 'fine': 'egg', 'coarse': '肉蛋生鲜',
         'bbox': (10, 10, 40, 40)}
        for _ in range(5)
    ])
    region = ChangedRegion((10, 10, 40, 40), 'SAME', before, after)
    events = det._analyze_regions([region])
    assert events == [('TAKE_OUT', {'removed': {19: 2}})]
    assert len(det.placed_items) == 3


def test_package_disappear_emits_package_take_out():
    det = EventDetector(_fake_motion, None)
    det.seed(STILL, [])
    region = ChangedRegion((10, 10, 40, 40), 'PACKAGE_DISAPPEAR',
                           {'name': '纯牛奶'}, None)
    events = det._analyze_regions([region])
    assert events == [('PACKAGE_TAKE_OUT',
                       {'removed_package': {'纯牛奶': 1}})]
