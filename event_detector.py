import cv2
from collections import Counter


class EventDetector:

    def __init__(self):
        self.state = 'IDLE'
        self.baseline_counts = {}
        self.stable_counts = {}
        self.stable_frames = 0
        self.idle_frames = 0
        self.IDLE_NEED = 10
        self.VOTE_FRAMES = 15  # 收集15帧
        self.vote_pool = []
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=60, detectShadows=False
        )

    def _to_counts(self, dets):
        c = Counter()
        for cls_id, conf, box in dets:
            c[int(cls_id)] += 1
        return dict(c)

    def _detect_motion(self, frame):
        fg = self.bg_sub.apply(frame)
        return cv2.countNonZero(fg) > 8000

    def _similar(self, a, b, tol=1):
        all_k = set(a.keys()) | set(b.keys())
        return all(abs(a.get(k, 0) - b.get(k, 0)) <= tol for k in all_k)

    def _total_diff(self, a, b):
        all_k = set(a.keys()) | set(b.keys())
        return sum(abs(a.get(k, 0) - b.get(k, 0)) for k in all_k)

    def compare_detections(self, before, after):
        all_cls = set(before.keys()) | set(after.keys())
        added, removed = {}, {}
        for cls in all_cls:
            b, a = before.get(cls, 0), after.get(cls, 0)
            if a > b:
                added[cls] = a - b
            elif a < b:
                removed[cls] = b - a

        if added and not removed:
            return 'PUT_IN', {'added': added}
        elif removed and not added:
            is_partial = any(before.get(c, 0) > n for c, n in removed.items())
            etype = 'PARTIAL_TAKE_OUT' if is_partial else 'TAKE_OUT'
            return etype, {'removed': removed}
        elif added and removed:
            return 'EXCHANGE', {'added': added, 'removed': removed}
        return 'NO_EVENT', {}

    def update(self, frame, current_dets):
        has_motion = self._detect_motion(frame)
        current_counts = self._to_counts(current_dets)
        event = None

        if self.state == 'IDLE':
            if self._similar(current_counts, self.stable_counts):
                self.idle_frames += 1
                if self.idle_frames >= self.IDLE_NEED:
                    self.baseline_counts = dict(self.stable_counts)
                    self.state = 'WATCHING'
                    self.idle_frames = 0
                    print(f'[事件] 开始监听，基准: {self.baseline_counts}')
            else:
                self.stable_counts = current_counts
                self.idle_frames = 0

        elif self.state == 'WATCHING':
            if self._total_diff(current_counts, self.baseline_counts) > 1:
                self.state = 'WAITING'
                self.stable_counts = current_counts
                self.stable_frames = 1
                print(f'[事件] 检测到变化: {self.baseline_counts} -> {current_counts}')


        elif self.state == 'WAITING':

            self.vote_pool.append(current_counts)

            if len(self.vote_pool) >= self.VOTE_FRAMES:

                # 找出现次数最多的counts

                str_pool = [str(sorted(c.items())) for c in self.vote_pool]

                winner_str = max(set(str_pool), key=str_pool.count)

                winner_count = str_pool.count(winner_str)

                # 从pool里找回对应的dict

                new_counts = {}

                for c in self.vote_pool:

                    if str(sorted(c.items())) == winner_str:
                        new_counts = c

                        break

                print(f'[投票] 众数: {new_counts} ({winner_count}/{self.VOTE_FRAMES}帧)')

                event_type, details = self.compare_detections(

                    self.baseline_counts, new_counts

                )

                print(f'[事件] 确认: {event_type} {details}')

                if event_type != 'NO_EVENT':
                    event = (event_type, details, current_dets)

                    self.baseline_counts = dict(new_counts)

                self.vote_pool = []

                self.state = 'IDLE'

                self.idle_frames = 0

        # 确保始终返回三元组
        return event, self.state, has_motion