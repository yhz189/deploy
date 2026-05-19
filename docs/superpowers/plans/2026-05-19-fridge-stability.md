# 冰箱食材识别系统 — 稳定性优化实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 用「变化定位」流水线替换不可靠的逐帧多目标计数差分，让冰箱食材识别在多物品场景下稳定、事件判断不误触发。

**架构：** 两态状态机（STABLE/BUSY）以运动门控；操作结束后对前后两张静止图做图像差分定位变化区域，只对裁剪出的单物体做识别，再映射到粗分类；维护物品位置记忆以支撑可靠的取出判断。

**技术栈：** Python 3.9、OpenCV、NumPy、RKNNLite（RKNN 1.4.0）、Flask、SQLite、pytest。

**对应规格：** `docs/superpowers/specs/2026-05-19-fridge-stability-design.md`

---

## 文件结构

| 文件 | 职责 | 改动 |
|------|------|------|
| `utils.py` | 预处理/后处理、粗类映射、裁剪识别 | 修改 |
| `motion.py` | 帧间运动判断 | 新建 |
| `change_locator.py` | 图像差分定位变化区域、区域识别 | 新建 |
| `event_detector.py` | 两态状态机、物品位置记忆、事件派生、整理交叉核对 | 重写 |
| `main.py` | 主循环接线、replay 模式 | 重写 |
| `web_server.py` | Web 界面，部分取出事件展示 | 小改 |
| `inventory.py` | SQLite 库存 | 不变 |
| `infer_camera.py` / `test_event.py` | 重复入口 / 损坏的测试 | 删除 |
| `tests/test_utils.py` | utils 单元测试 | 新建 |
| `tests/test_motion.py` | motion 单元测试 | 新建 |
| `tests/test_change_locator.py` | change_locator 单元测试 | 新建 |
| `tests/test_event_detector.py` | event_detector 单元测试 | 新建 |

**统一接口约定（贯穿全计划）：**

- `classify_fn(crop_bgr) -> dict | None`：识别一张裁剪图，返回 `{'class_id': int, 'fine': str, 'coarse': str, 'conf': float, 'area': float}`，识别不出返回 `None`。
- `ChangedRegion(bbox, kind, ref_ident, new_ident)`：`bbox` 为 `(x, y, w, h)`；`kind ∈ {'APPEAR','DISAPPEAR','REPLACE','SAME'}`；`ref_ident`/`new_ident` 为 `classify_fn` 的返回值。
- 事件元组：`('PUT_IN', {'added': {class_id: qty}})`、`('TAKE_OUT', {'removed': {class_id: qty}})`、`('PARTIAL_TAKE_OUT', {'removed': {class_id: qty}})`。
- 物品位置记忆记录：`{'id': int, 'class_id': int, 'fine': str, 'coarse': str, 'bbox': (x,y,w,h)}`。

**范围说明：** 规格 §6 提到 Web「待确认」交互。本计划实现到「部分取出事件带估计值并在 Web 上以徽章区分展示」为止；用户点击修正数量的交互式端点需要 `inventory` 加表字段，留作后续加分项，不在本计划内（与规格 §4.3「`inventory.py` 不变」保持一致）。

---

## 任务 1：测试环境与粗类映射

**文件：**
- 创建：`tests/__init__.py`（空文件）
- 修改：`utils.py`（新增 `COARSE_MAP` 与 `to_coarse`）
- 测试：`tests/test_utils.py`

- [ ] **步骤 1：准备测试环境**

在开发机（Windows）执行，确认 pytest 可用：

```bash
pip install pytest
```

创建空文件 `tests/__init__.py`（内容为空，使 tests 成为包）。

- [ ] **步骤 2：编写失败的测试**

创建 `tests/test_utils.py`：

```python
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
```

- [ ] **步骤 3：运行测试验证失败**

运行：`pytest tests/test_utils.py -v`
预期：FAIL，报错 `ImportError: cannot import name 'to_coarse'`

- [ ] **步骤 4：实现粗类映射**

在 `utils.py` 中，`CLASSES` 定义之后新增：

```python
COARSE_MAP = {
    'Apple': '蔬果', 'Avocado': '蔬果', 'Banana': '蔬果',
    'Bell pepper': '蔬果', 'Broccoli': '蔬果', 'Carrot': '蔬果',
    'Garlic': '蔬果', 'Lemon': '蔬果', 'Tomato': '蔬果',
    'Chicken': '生鲜', 'Eggs': '生鲜',
    'Butter': '乳品', 'Cheese': '乳品', 'Cooking cream': '乳品',
    'Bread': '包装食品', 'Hot Sauce': '包装食品', 'Ketchup': '包装食品',
}


def to_coarse(class_id):
    """细类 id → 粗分类名称"""
    return COARSE_MAP[CLASSES[class_id]]
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_utils.py -v`
预期：3 个测试全部 PASS

- [ ] **步骤 6：Commit**

```bash
git add tests/__init__.py tests/test_utils.py utils.py
git commit -m "feat: 新增 17 细类到 4 粗类的映射"
```

---

## 任务 2：修复 utils.py 的 NMS 与置信度阈值

**文件：**
- 修改：`utils.py:12`（`CONF_THRESH`）、`utils.py:81-91`（`postprocess` 的 NMS）
- 测试：`tests/test_utils.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_utils.py` 末尾追加：

```python
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
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_utils.py::test_postprocess_nms_dedups_overlapping_boxes -v`
预期：FAIL —— 当前 `postprocess` 把 xyxy 直接传给 `cv2.dnn.NMSBoxes`（要求 xywh），IoU 计算错误，去重结果不为 2。

- [ ] **步骤 3：修复 CONF_THRESH 与 NMS 坐标格式**

`utils.py:12` 改为：

```python
CONF_THRESH = 0.30
```

`utils.py` 中 `postprocess` 的 NMS 部分，将：

```python
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), confidences.tolist(), CONF_THRESH, NMS_THRESH
    )
```

改为：

```python
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS：cv2.dnn.NMSBoxes 要求 [x, y, w, h] 格式，需从 xyxy 转换
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), confidences.tolist(), CONF_THRESH, NMS_THRESH
    )
```

后续 `indices` 仍用于索引 xyxy 的 `boxes`，无需改动。

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_utils.py -v`
预期：全部 PASS（含任务 1 的 3 个）

- [ ] **步骤 5：Commit**

```bash
git add utils.py tests/test_utils.py
git commit -m "fix: 修复 NMS 坐标格式错误并下调置信度阈值"
```

---

## 任务 3：运动门控模块 motion.py

**文件：**
- 创建：`motion.py`
- 测试：`tests/test_motion.py`

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_motion.py`：

```python
import numpy as np
from motion import is_moving


def test_identical_frames_not_moving():
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    assert is_moving(frame, frame) is False


def test_large_change_is_moving():
    prev = np.full((480, 640, 3), 120, dtype=np.uint8)
    cur = prev.copy()
    cur[100:300, 100:400] = 255  # 一大块区域明显变化
    assert is_moving(prev, cur) is True


def test_tiny_noise_not_moving():
    prev = np.full((480, 640, 3), 120, dtype=np.uint8)
    cur = prev.copy()
    cur[0:5, 0:5] = 255  # 极小噪点，不算运动
    assert is_moving(prev, cur) is False
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_motion.py -v`
预期：FAIL，`ModuleNotFoundError: No module named 'motion'`

- [ ] **步骤 3：实现 motion.py**

创建 `motion.py`：

```python
"""运动门控：判断两帧之间是否有运动"""
import cv2


def is_moving(prev_bgr, cur_bgr, diff_thresh=25, area_thresh=3000):
    """两帧灰度差分，运动像素数超过 area_thresh 即判定有运动"""
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, cur_gray)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    return int(cv2.countNonZero(mask)) > area_thresh
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_motion.py -v`
预期：3 个测试全部 PASS

- [ ] **步骤 5：Commit**

```bash
git add motion.py tests/test_motion.py
git commit -m "feat: 新增帧间运动门控模块"
```

---

## 任务 4：变化区域定位 change_locator.find_change_regions

**文件：**
- 创建：`change_locator.py`
- 测试：`tests/test_change_locator.py`

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_change_locator.py`：

```python
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
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_change_locator.py -v`
预期：FAIL，`ModuleNotFoundError: No module named 'change_locator'`

- [ ] **步骤 3：实现 find_change_regions**

创建 `change_locator.py`：

```python
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
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_change_locator.py -v`
预期：3 个测试全部 PASS

- [ ] **步骤 5：Commit**

```bash
git add change_locator.py tests/test_change_locator.py
git commit -m "feat: 新增图像差分变化区域定位"
```

---

## 任务 5：变化区域识别 change_locator.classify_regions

**文件：**
- 修改：`change_locator.py`（新增 `ChangedRegion` 与 `classify_regions`）
- 测试：`tests/test_change_locator.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_change_locator.py` 末尾追加：

```python
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


def test_classify_noise_dropped():
    ref = np.zeros((100, 100, 3), dtype=np.uint8)
    new = np.zeros((100, 100, 3), dtype=np.uint8)
    classify_fn = _fake_classify({0: None})  # 两侧都识别不出
    regions = classify_regions(ref, new, [(0, 0, 50, 50)], classify_fn)
    assert regions == []
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_change_locator.py -v`
预期：新增 3 个测试 FAIL，`ImportError: cannot import name 'classify_regions'`

- [ ] **步骤 3：实现 ChangedRegion 与 classify_regions**

在 `change_locator.py` 的 `import` 之后、`find_change_regions` 之前新增：

```python
@dataclass
class ChangedRegion:
    bbox: tuple                 # (x, y, w, h)
    kind: str                   # APPEAR | DISAPPEAR | REPLACE | SAME
    ref_ident: dict             # classify_fn 对 ref 裁剪图的结果，可为 None
    new_ident: dict             # classify_fn 对 new 裁剪图的结果，可为 None
```

在文件末尾新增：

```python
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
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_change_locator.py -v`
预期：6 个测试全部 PASS

- [ ] **步骤 5：Commit**

```bash
git add change_locator.py tests/test_change_locator.py
git commit -m "feat: 新增变化区域识别与 APPEAR/DISAPPEAR/REPLACE 判定"
```

---

## 任务 6：裁剪图识别 utils.identify_crop

**文件：**
- 修改：`utils.py`（新增 `identify_crop`）
- 测试：`tests/test_utils.py`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_utils.py` 末尾追加：

```python
from utils import identify_crop


class _FakeModel:
    """模拟 RKNNLite：inference 返回预置的 outputs"""
    def __init__(self, outputs):
        self._outputs = outputs

    def inference(self, inputs):
        return self._outputs


def test_identify_crop_returns_best_detection():
    crop = np.full((80, 80, 3), 120, dtype=np.uint8)
    outputs = _make_outputs(
        boxes_xyxy=[(10, 10, 60, 60)], class_id=0, scores=[0.85])
    model = _FakeModel(outputs)
    ident = identify_crop(model, crop)
    assert ident is not None
    assert ident['class_id'] == 0
    assert ident['fine'] == 'Apple'
    assert ident['coarse'] == '蔬果'
    assert ident['area'] > 0


def test_identify_crop_empty_returns_none():
    model = _FakeModel(None)
    assert identify_crop(model, None) is None
    assert identify_crop(model, np.zeros((0, 0, 3), dtype=np.uint8)) is None
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_utils.py -v`
预期：新增 2 个测试 FAIL，`ImportError: cannot import name 'identify_crop'`

- [ ] **步骤 3：实现 identify_crop**

在 `utils.py` 末尾新增（依赖已有的 `preprocess`、`postprocess`、`CLASSES` 与任务 1 的 `to_coarse`）：

```python
def identify_crop(model, crop_bgr):
    """对一张裁剪图运行识别，返回置信度最高的检测结果或 None"""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    img_input, ratio, pad = preprocess(crop_bgr)
    outputs = model.inference(inputs=[img_input])
    if outputs is None or outputs[0] is None:
        return None
    boxes, confs, class_ids = postprocess(
        outputs, ratio, pad, crop_bgr.shape)
    if len(boxes) == 0:
        return None
    best = int(np.argmax(confs))
    cid = int(class_ids[best])
    x1, y1, x2, y2 = boxes[best]
    return {
        'class_id': cid,
        'fine': CLASSES[cid],
        'coarse': to_coarse(cid),
        'conf': float(confs[best]),
        'area': float(abs((x2 - x1) * (y2 - y1))),
    }
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_utils.py -v`
预期：全部 PASS

- [ ] **步骤 5：Commit**

```bash
git add utils.py tests/test_utils.py
git commit -m "feat: 新增裁剪图单物体识别函数"
```

---

## 任务 7：重写 event_detector.py 状态机

**文件：**
- 重写：`event_detector.py`
- 测试：`tests/test_event_detector.py`

- [ ] **步骤 1：编写状态机流转的失败测试**

创建 `tests/test_event_detector.py`：

```python
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
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_event_detector.py -v`
预期：FAIL —— 旧 `EventDetector` 构造签名不符（无 `motion_fn` 参数）。

- [ ] **步骤 3：实现 EventDetector 状态机骨架**

将 `event_detector.py` 整个文件替换为：

```python
"""事件检测器：两态状态机 + 物品位置记忆 + 变化定位事件派生"""

PARTIAL_AREA_RATIO = 0.7
IOU_MATCH_THRESH = 0.3
SIZE_MATCH_TOLERANCE = 0.3


def _iou(b1, b2):
    """两个 (x, y, w, h) 框的 IoU"""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def _size_similar(b1, b2):
    a1, a2 = b1[2] * b1[3], b2[2] * b2[3]
    if max(a1, a2) == 0:
        return False
    return abs(a1 - a2) / max(a1, a2) <= SIZE_MATCH_TOLERANCE


class EventDetector:

    def __init__(self, motion_fn, locate_fn,
                 enter_frames=3, exit_frames=10, settle_frames=5):
        self.motion_fn = motion_fn      # (prev_bgr, cur_bgr) -> bool
        self.locate_fn = locate_fn      # (ref_bgr, new_bgr) -> list[ChangedRegion]
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.settle_frames = settle_frames

        self.state = 'STABLE'
        self.prev_frame = None
        self.ref_frame = None
        self.placed_items = []
        self._next_id = 1
        self._moving_streak = 0
        self._still_streak = 0
        self._settle_streak = 0
        self._settle_frame = None

    def seed(self, frame, detections):
        """开机播种：detections 为 [{'class_id','fine','coarse','bbox'}, ...]"""
        self.ref_frame = frame.copy()
        self.prev_frame = frame.copy()
        for d in detections:
            self._add_item(d['class_id'], d['fine'], d['coarse'], d['bbox'])

    def _add_item(self, class_id, fine, coarse, bbox):
        rec = {'id': self._next_id, 'class_id': class_id,
               'fine': fine, 'coarse': coarse, 'bbox': bbox}
        self.placed_items.append(rec)
        self._next_id += 1
        return rec

    def update(self, frame):
        """喂入一帧，返回 (events, state)。events 为事件元组列表。"""
        events = []
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            self.ref_frame = frame.copy()
            return events, self.state

        moving = self.motion_fn(self.prev_frame, frame)
        self.prev_frame = frame.copy()

        if self.state == 'STABLE':
            if moving:
                self._moving_streak += 1
                if self._moving_streak >= self.enter_frames:
                    self.state = 'BUSY'
                    self._moving_streak = 0
                    self._still_streak = 0
            else:
                self._moving_streak = 0
                self.ref_frame = frame.copy()  # 静止期持续刷新参考图

        elif self.state == 'BUSY':
            if moving:
                self._still_streak = 0
            else:
                self._still_streak += 1
                if self._still_streak >= self.exit_frames:
                    self.state = 'SETTLING'
                    self._settle_streak = 0
                    self._settle_frame = frame.copy()

        elif self.state == 'SETTLING':
            if moving:
                self.state = 'BUSY'
                self._still_streak = 0
            else:
                self._settle_streak += 1
                if self._settle_streak >= self.settle_frames:
                    events = self._analyze(self._settle_frame)
                    self.ref_frame = self._settle_frame.copy()
                    self.state = 'STABLE'

        return events, self.state

    def _analyze(self, new_frame):
        """对 ref_frame 与 new_frame 做变化分析，返回事件列表。"""
        return []  # 任务 7 步骤 6 实现
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_event_detector.py -v`
预期：3 个状态机测试全部 PASS

- [ ] **步骤 5：编写事件派生的失败测试**

在 `tests/test_event_detector.py` 末尾追加：

```python
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
```

- [ ] **步骤 6：运行测试验证失败**

运行：`pytest tests/test_event_detector.py -v`
预期：新增 4 个测试 FAIL，`AttributeError: '_analyze_regions'`，且 `_analyze` 当前返回空。

- [ ] **步骤 7：实现事件派生逻辑**

将 `event_detector.py` 中 `_analyze` 方法替换为以下完整实现：

```python
    def _analyze(self, new_frame):
        regions = self.locate_fn(self.ref_frame, new_frame)
        return self._analyze_regions(regions)

    def _analyze_regions(self, regions):
        events = []
        appears, disappears = [], []
        for r in regions:
            if r.kind == 'APPEAR':
                appears.append(r)
            elif r.kind == 'DISAPPEAR':
                disappears.append(r)
            elif r.kind == 'REPLACE':
                events += self._take_out(r.bbox, r.ref_ident)
                events += self._put_in(r.bbox, r.new_ident)
            elif r.kind == 'SAME':
                events += self._handle_same(r)
        events += self._cross_check(appears, disappears)
        return events

    def _cross_check(self, appears, disappears):
        """整理交叉核对：同粗类、尺寸相近的一出一进配对为「整理」，不计事件"""
        events = []
        unmatched = list(disappears)
        for a in appears:
            match = None
            for d in unmatched:
                if (a.new_ident['coarse'] == d.ref_ident['coarse']
                        and _size_similar(a.bbox, d.bbox)):
                    match = d
                    break
            if match is not None:
                unmatched.remove(match)
                self._move_item(match.bbox, a.bbox)  # 整理：更新位置记忆
            else:
                events += self._put_in(a.bbox, a.new_ident)
        for d in unmatched:
            events += self._take_out(d.bbox, d.ref_ident)
        return events

    def _put_in(self, bbox, ident):
        self._add_item(ident['class_id'], ident['fine'],
                       ident['coarse'], bbox)
        return [('PUT_IN', {'added': {ident['class_id']: 1}})]

    def _take_out(self, bbox, ref_ident):
        rec = self._match_item(bbox)
        if rec is None:
            return []  # 记忆里没有对应物品，不误更新库存
        self.placed_items.remove(rec)
        return [('TAKE_OUT', {'removed': {rec['class_id']: 1}})]

    def _handle_same(self, r):
        """同位置同粗类：按检测框面积判断部分取出 / 追加 / 位置抖动"""
        ra = r.ref_ident['area']
        na = r.new_ident['area']
        if ra <= 0:
            return []
        if na < ra * PARTIAL_AREA_RATIO:
            rec = self._match_item(r.bbox)
            cid = rec['class_id'] if rec else r.ref_ident['class_id']
            return [('PARTIAL_TAKE_OUT', {'removed': {cid: 1}})]
        if na > ra / PARTIAL_AREA_RATIO:
            return self._put_in(r.bbox, r.new_ident)
        return []  # 面积相当，判为位置抖动，忽略

    def _match_item(self, bbox):
        """按 IoU 在位置记忆里找最匹配的物品"""
        best, best_iou = None, IOU_MATCH_THRESH
        for rec in self.placed_items:
            iou = _iou(rec['bbox'], bbox)
            if iou >= best_iou:
                best, best_iou = rec, iou
        return best

    def _move_item(self, old_bbox, new_bbox):
        rec = self._match_item(old_bbox)
        if rec is not None:
            rec['bbox'] = new_bbox
```

- [ ] **步骤 8：运行测试验证通过**

运行：`pytest tests/test_event_detector.py -v`
预期：7 个测试全部 PASS

- [ ] **步骤 9：Commit**

```bash
git add event_detector.py tests/test_event_detector.py
git commit -m "feat: 重写事件检测器为两态状态机 + 变化定位事件派生"
```

---

## 任务 8：重写 main.py 主程序

**文件：**
- 重写：`main.py`
- 测试：无自动化测试（涉及 RKNN/摄像头，由任务 11 板端验证）

- [ ] **步骤 1：重写 main.py**

将 `main.py` 整个文件替换为：

```python
"""冰箱食材识别与管理系统 - 主程序

实时模式：  python main.py
回放模式：  python main.py --replay results/clip   （喂帧序列目录调试）
"""
import os
import sys
import glob
import time

import cv2
from rknnlite.api import RKNNLite

from utils import preprocess, postprocess, identify_crop, to_coarse, CLASSES
from motion import is_moving
from change_locator import find_change_regions, classify_regions
from event_detector import EventDetector
from inventory import InventoryManager

RKNN_MODEL = 'models/fridge_yolo_fp16.rknn'
CAMERA_ID = 0


def open_source():
    """返回一个逐帧产出 BGR 图的生成器；支持实时摄像头与回放目录"""
    if len(sys.argv) >= 3 and sys.argv[1] == '--replay':
        paths = sorted(glob.glob(os.path.join(sys.argv[2], '*.jpg')))
        assert paths, f'回放目录无 jpg：{sys.argv[2]}'
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                yield img
    else:
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        assert cap.isOpened(), '摄像头打开失败'
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                yield frame
        finally:
            cap.release()


def seed_detections(model, frame):
    """开机全画面检测一次，给位置记忆与库存播种"""
    img_input, ratio, pad = preprocess(frame)
    outputs = model.inference(inputs=[img_input])
    if outputs is None or outputs[0] is None:
        return []
    boxes, confs, class_ids = postprocess(outputs, ratio, pad, frame.shape)
    dets = []
    for box, cid in zip(boxes, class_ids):
        x1, y1, x2, y2 = box
        dets.append({'class_id': int(cid), 'fine': CLASSES[int(cid)],
                     'coarse': to_coarse(int(cid)),
                     'bbox': (int(x1), int(y1),
                              int(x2 - x1), int(y2 - y1))})
    return dets


def main():
    model = RKNNLite()
    model.load_rknn(RKNN_MODEL)
    model.init_runtime()
    print('✓ 模型加载成功')

    inv = InventoryManager()

    def classify_fn(crop):
        return identify_crop(model, crop)

    def locate_fn(ref, new):
        bboxes = find_change_regions(ref, new)
        return classify_regions(ref, new, bboxes, classify_fn)

    detector = EventDetector(is_moving, locate_fn)
    print('✓ 事件检测器就绪')

    frames = open_source()
    first = next(frames, None)
    assert first is not None, '无可用帧'

    dets = seed_detections(model, first)
    detector.seed(first, dets)
    for d in dets:
        inv.process_event('PUT_IN', {'added': {d['class_id']: 1}})
    print(f'✓ 开机播种：{len(dets)} 个物品')
    inv.print_stock()

    frame_count = 0
    try:
        for frame in frames:
            frame_count += 1
            events, state = detector.update(frame)
            for event_type, details in events:
                inv.process_event(event_type, details)
                print(f'[事件] {event_type} {details}')
                inv.print_stock()
            if frame_count % 30 == 0:
                print(f'帧{frame_count:5d} | 状态:{state}')
    except KeyboardInterrupt:
        print('\n用户中断')
    finally:
        print('\n====== 最终库存 ======')
        inv.print_stock()
        inv.close()
        model.release()
        print('系统退出')


if __name__ == '__main__':
    main()
```

- [ ] **步骤 2：本地静态检查**

在开发机运行（无 RKNN 环境会在 import 处报错，这是预期的，仅检查语法）：

```bash
python -c "import ast; ast.parse(open('main.py', encoding='utf-8').read()); print('语法 OK')"
```

预期：输出 `语法 OK`

- [ ] **步骤 3：Commit**

```bash
git add main.py
git commit -m "feat: 重写主程序接入变化定位流水线并支持回放模式"
```

---

## 任务 9：web_server.py 部分取出事件展示

**文件：**
- 修改：`web_server.py`（事件表区分部分取出）

- [ ] **步骤 1：修改事件徽章渲染**

`web_server.py` 的 `HTML` 模板中，事件表的徽章逻辑当前为：

```html
                <td>
                    {% if 'PUT_IN' in etype %}
                    <span class="badge">放入</span>
                    {% else %}
                    <span class="badge badge-out">取出</span>
                    {% endif %}
                </td>
```

替换为：

```html
                <td>
                    {% if 'PUT_IN' in etype %}
                    <span class="badge">放入</span>
                    {% elif 'PARTIAL' in etype %}
                    <span class="badge badge-partial">部分取出·估计</span>
                    {% else %}
                    <span class="badge badge-out">取出</span>
                    {% endif %}
                </td>
```

在 `<style>` 块中 `.badge-out` 一行之后新增：

```css
        .badge-partial { background:#ff9800; }
```

- [ ] **步骤 2：本地静态检查**

```bash
python -c "import ast; ast.parse(open('web_server.py', encoding='utf-8').read()); print('语法 OK')"
```

预期：输出 `语法 OK`

- [ ] **步骤 3：Commit**

```bash
git add web_server.py
git commit -m "feat: Web 界面区分展示部分取出事件"
```

---

## 任务 10：清理冗余文件

**文件：**
- 删除：`infer_camera.py`、`test_event.py`

- [ ] **步骤 1：删除冗余文件**

```bash
git rm infer_camera.py test_event.py
```

`infer_camera.py` 是与 `main.py` 重复的旧入口；`test_event.py` 误用了 `compare_detections` 接口（传检测列表而非计数字典），已由 `tests/` 下的新单元测试替代。

- [ ] **步骤 2：确认全部单元测试仍通过**

运行：`pytest tests/ -v`
预期：任务 1-7 的全部测试 PASS（共 22 个）

- [ ] **步骤 3：Commit**

```bash
git commit -m "chore: 删除重复入口与损坏的旧测试"
```

---

## 任务 11：板端集成验证

**文件：** 无代码改动，本任务为开发板上的人工验证清单。

- [ ] **步骤 1：同步代码到开发板**

将 `utils.py`、`motion.py`、`change_locator.py`、`event_detector.py`、`main.py`、`web_server.py`、`inventory.py`、`models/` 拷贝到开发板。

- [ ] **步骤 2：录制一段回放素材**

在板子上用摄像头录一段「静止 → 放入一个食材 → 静止」的帧序列保存到目录（可临时写脚本每帧 `cv2.imwrite`），用于可重复调参。

- [ ] **步骤 3：回放模式跑通流水线**

运行：`python main.py --replay <素材目录>`
预期：日志出现一次 `[事件] PUT_IN`，库存随之更新；状态在 STABLE/BUSY/SETTLING 间正常流转。

- [ ] **步骤 4：按规格 §11 调参**

若误触发或漏触发，依据规格 `docs/superpowers/specs/2026-05-19-fridge-stability-design.md` §11 参数表，在板上实测调整 `motion.is_moving` 的 `area_thresh`、`find_change_regions` 的 `diff_thresh`/`min_area`、`EventDetector` 的 `enter/exit/settle_frames`。

- [ ] **步骤 5：实时模式验证四类场景**

运行 `python main.py`（另开终端 `python web_server.py`），依次验证规格要求的演示场景，确认库存与 Web 界面正确：
1. 正常放入一个食材 → `PUT_IN`
2. 正常取出一个食材 → `TAKE_OUT`
3. 部分取出（一袋拿走一部分）→ `PARTIAL_TAKE_OUT`，Web 显示「部分取出·估计」徽章
4. 仅伸手整理、不增减物品 → 无事件（库存不变）

- [ ] **步骤 6：Commit 调参结果**

```bash
git add motion.py change_locator.py event_detector.py
git commit -m "chore: 按板端实测结果调整门控与差分参数"
```

---

## 自检结果

**规格覆盖度：**
- §3 方案选择 → 任务 4-7 整体实现变化定位流水线
- §4.1 两态状态机 → 任务 7
- §4.3 模块划分 → 任务 1-10 文件结构一一对应
- §5.1 motion → 任务 3
- §5.2 change_locator → 任务 4、5
- §5.3 物品位置记忆 → 任务 7（`placed_items`、`_match_item`、`seed`）
- §5.4 分层识别 → 任务 1（`to_coarse`）、任务 6（`identify_crop` 返回 `coarse`）
- §5.5 事件派生 + 交叉核对 → 任务 7（`_analyze_regions`、`_cross_check`）
- §6 部分取出 → 任务 7（`_handle_same`）、任务 9（Web 展示）
- §7 边界情况 → 整理(任务7 `_cross_check`)、噪声(任务5)、开机播种(任务8 `seed_detections`)、采样时机(任务7 `SETTLING`+`settle_frames`)
- §8 Bug 修复 → 任务 2（NMS、CONF_THRESH）；空检测污染/运动门控未接入/旧状态名 → 任务 7、8 重写消除；`test_event.py` → 任务 10 删除
- §9 错误处理 → 任务 6（裁剪空返回 None）、任务 7（`_take_out` 记忆未命中不更新）、任务 8（推理返回 None 跳过）
- §10 测试方案 → 任务 1-7 的 `tests/`，replay 模式 → 任务 8
- §11 关键参数 → 各模块默认参数，任务 11 板端实测调整

**占位符扫描：** 无「待定/TODO」；任务 7 步骤 3 的 `_analyze` 临时返回空有注释说明，并在步骤 7 替换为完整实现。

**类型一致性：** `classify_fn` 返回字典键（`class_id/fine/coarse/conf/area`）、`ChangedRegion` 字段、事件元组结构、`placed_items` 记录结构在任务 5/6/7 间一致；`EventDetector(motion_fn, locate_fn, ...)` 构造签名在任务 7、8 与测试中一致。
