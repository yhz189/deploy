from utils import to_coarse, CLASSES


def test_to_coarse_dairy():
    # Butter / Cheese / Cooking cream 都应归入「乳品」
    butter_id = CLASSES.index('Butter')
    cheese_id = CLASSES.index('Cheese')
    assert to_coarse(butter_id) == '乳品'
    assert to_coarse(cheese_id) == '乳品'


def test_to_coarse_produce():
    assert to_coarse(CLASSES.index('Apple')) == '蔬果'
    assert to_coarse(CLASSES.index('Tomato')) == '蔬果'


def test_to_coarse_covers_all_classes():
    # 17 个细类每个都要有粗类映射
    for cid in range(len(CLASSES)):
        assert to_coarse(cid) in ('蔬果', '生鲜', '乳品', '包装食品')


import numpy as np
from utils import postprocess


def _make_outputs(boxes_xyxy, class_id, scores):
    """构造 (1, 21, N, 1) 的模型输出：前 4 行 xyxy，后 17 行类别得分"""
    n = len(boxes_xyxy)
    pred = np.zeros((1, 21, n, 1), dtype=np.float32)
    for i, (box, score) in enumerate(zip(boxes_xyxy, scores)):
        pred[0, 0:4, i, 0] = box
        pred[0, 4 + class_id, i, 0] = score
    return [pred]


def test_postprocess_nms_dedups_overlapping_boxes():
    # 两个几乎重合的同类框 + 一个独立框 → NMS 后应剩 2 个
    outputs = _make_outputs(
        boxes_xyxy=[(100, 100, 150, 150),
                    (105, 105, 155, 155),
                    (300, 300, 360, 360)],
        class_id=0,
        scores=[0.9, 0.8, 0.85],
    )
    boxes, confs, class_ids = postprocess(
        outputs, ratio=1.0, pad=(0, 0), orig_shape=(480, 640, 3))
    assert len(boxes) == 2
    assert all(c == 0 for c in class_ids)
