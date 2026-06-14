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
                 enter_frames=3, exit_frames=10, settle_frames=12):
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
        self.last_regions = []          # 最近一次分析的变化区域，供可视化读取

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
                self._settle_frame = frame.copy()
                if self._settle_streak >= self.settle_frames:
                    events = self._analyze(self._settle_frame)
                    self.ref_frame = self._settle_frame.copy()
                    self.state = 'STABLE'

        return events, self.state

    def _analyze(self, new_frame):
        regions = self.locate_fn(self.ref_frame, new_frame)
        self.last_regions = regions
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
            elif r.kind == 'NOISE':
                rec = self._match_item(r.bbox)
                if rec is not None:
                    self.placed_items.remove(rec)
                    events.append(('TAKE_OUT', {'removed': {rec['class_id']: 1}}))
            elif r.kind == 'PACKAGE_DISAPPEAR':
                name = r.ref_ident.get('name')
                if name:
                    events.append(('PACKAGE_TAKE_OUT',
                                   {'removed_package': {name: 1}}))
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
        count = ident.get('count', 1)
        for _ in range(count):
            self._add_item(ident['class_id'], ident['fine'],
                           ident['coarse'], bbox)
        return [('PUT_IN', {'added': {ident['class_id']: count}})]

    def _take_out(self, bbox, ref_ident):
        if ref_ident is not None:
            count = ref_ident.get('count', 1)
            removed = self._remove_items(
                bbox, ref_ident['class_id'], count)
            if removed:
                return [('TAKE_OUT', {'removed': {ref_ident['class_id']: removed}})]
            # 记忆未命中：回退用模型识别出的旧物品类别，避免吞掉出库事件
            return [('TAKE_OUT', {'removed': {ref_ident['class_id']: count}})]

        rec = self._match_item(bbox)
        if rec is not None:
            self.placed_items.remove(rec)
            return [('TAKE_OUT', {'removed': {rec['class_id']: 1}})]
        return []

    def _handle_same(self, r):
        """同位置同粗类：按检测框面积判断部分取出 / 追加 / 位置抖动"""
        if r.ref_ident['class_id'] == r.new_ident['class_id']:
            cid = r.ref_ident['class_id']
            ref_count = r.ref_ident.get('count', 1)
            new_count = r.new_ident.get('count', 1)
            if new_count < ref_count:
                delta = ref_count - new_count
                removed = self._remove_items(r.bbox, cid, delta)
                return [('TAKE_OUT', {'removed': {cid: removed or delta}})]
            if new_count > ref_count:
                ident = dict(r.new_ident)
                ident['count'] = new_count - ref_count
                return self._put_in(r.bbox, ident)

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

    def _remove_items(self, bbox, class_id, count):
        removed = 0
        for _ in range(count):
            rec = self._match_item(bbox)
            if rec is None or rec['class_id'] != class_id:
                break
            self.placed_items.remove(rec)
            removed += 1
        return removed
