"""冰箱食材识别与管理系统 - 主程序

实时模式：       python main.py
实时 + 显示窗口： python main.py --show
回放模式：       python main.py --replay results/clip
回放 + 显示窗口： python main.py --replay results/clip --show

--show 会在板子接的 HDMI 显示器上开一个实时窗口，叠加显示状态机状态、
变化区域框、最近事件。窗口里按 q 退出。
（注意：--show 需在板子本地桌面运行，或 SSH 时加 DISPLAY=:0）
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
from package_ocr import (save_package_candidate,
                         save_package_takeout_candidate)

RKNN_MODEL = 'models/fridge_yolo_v3.rknn'
CAMERA_ID = 0
PREVIEW_PATH = '/tmp/fridge_latest.jpg'
PREVIEW_INTERVAL = 30          # 每30帧保存一次预览（约1秒）
REGION_SHOW_FRAMES = 90        # 分析结束后变化框持续显示的帧数
PACKAGE_OCR_CONF_TRIGGER = 0.35  # YOLO 低置信度时尝试 OCR 辅助包装建档

SHOW = '--show' in sys.argv    # 是否开实时显示窗口

STATE_COLORS = {
    'STABLE':   (0, 200, 0),
    'BUSY':     (0, 140, 255),
    'SETTLING': (255, 140, 0),
}

REGION_COLORS = {
    'APPEAR':    (0, 200, 0),
    'DISAPPEAR': (0, 0, 255),
    'REPLACE':   (255, 0, 255),
    'SAME':      (200, 200, 0),
    'NOISE':     (130, 130, 130),
}


def format_event(event_type, details):
    """事件格式化成 ASCII 字符串（cv2.putText 不支持中文）"""
    parts = []
    for key in ('added', 'removed'):
        for cid, n in details.get(key, {}).items():
            parts.append(f'{CLASSES[cid]}x{n}')
    return f"{event_type} {' '.join(parts)}".strip()


def draw_overlay(frame, state, frame_count, last_event='', regions=None):
    """在帧上叠加状态/帧号/事件/变化区域，返回新图（不改原图）"""
    img = frame.copy()
    if regions:
        for r in regions:
            x, y, w, h = r.bbox
            rc = REGION_COLORS.get(r.kind, (200, 200, 200))
            cv2.rectangle(img, (x, y), (x + w, y + h), rc, 2)
            ident = r.new_ident or r.ref_ident
            ident_name = None if ident is None else (
                ident.get('fine') or ident.get('name') or ident.get('category'))
            label = r.kind if not ident_name else f'{r.kind}:{ident_name}'
            cv2.putText(img, label, (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, rc, 1, cv2.LINE_AA)
    color = STATE_COLORS.get(state, (128, 128, 128))
    cv2.putText(img, state, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(img, f'frame {frame_count}', (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    if last_event:
        cv2.putText(img, last_event, (10, img.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return img


def save_preview(frame, state, frame_count=0, last_event=''):
    """保存预览图给 Web 端 /camera 路由读取"""
    img = draw_overlay(frame, state, frame_count, last_event)
    cv2.imwrite(PREVIEW_PATH, img)


def maybe_save_package_candidate(new_frame, regions):
    """保存疑似包装物品裁剪图，交给手机 App 做本地 OCR。"""
    candidates = []
    for r in regions:
        if r.kind == 'NOISE':
            candidates.append(r)
        elif r.kind == 'APPEAR' and r.new_ident is not None:
            if r.new_ident.get('conf', 1.0) < PACKAGE_OCR_CONF_TRIGGER:
                candidates.append(r)
    if not candidates:
        return None

    # 选最大变化区域做 OCR，避免一帧内对多个小噪声重复识别。
    region = max(candidates, key=lambda rr: rr.bbox[2] * rr.bbox[3])
    x, y, w, h = region.bbox
    crop = new_frame[y:y + h, x:x + w]
    result = {
        'ok': False,
        'engine': 'phone_ocr',
        'error': 'pending phone OCR',
        'name': '',
        'confidence': 0.0,
        'raw_text': [],
    }
    cand = save_package_candidate(crop, result, bbox=region.bbox)
    # 低置信度 APPEAR 更可能是包装/非训练类物品，避免误入普通生鲜库存。
    if region.kind == 'APPEAR':
        region.kind = 'NOISE'
    print('[OCR] 已保存疑似包装裁剪图，等待手机端 OCR 后提交入库')
    return cand


def maybe_save_package_takeout_candidate(ref_frame, new_frame, regions):
    """保存非生鲜变化区域的前后裁剪图，供手机 OCR 判断包装取出。"""
    candidates = [r for r in regions if r.kind == 'NOISE']
    if not candidates:
        return None

    region = max(candidates, key=lambda rr: rr.bbox[2] * rr.bbox[3])
    x, y, w, h = region.bbox
    ref_crop = ref_frame[y:y + h, x:x + w]
    new_crop = new_frame[y:y + h, x:x + w]
    cand = save_package_takeout_candidate(
        ref_crop, new_crop, bbox=region.bbox, reason='noise_region')
    if cand is not None:
        print('[OCR] 已保存包装取出候选前后图，等待手机端 OCR 判断')
    return cand


def mark_package_take_out(regions, inv):
    """把与已入库包装物品位置重合的 NOISE 区域标成包装取出事件。"""
    for r in regions:
        if r.kind != 'NOISE':
            continue
        match = inv.match_package_by_bbox(r.bbox)
        if match is not None:
            r.kind = 'PACKAGE_DISAPPEAR'
            r.ref_ident = {'name': match['name'], 'bbox': match['bbox']}
            print(f"[包装取出] 匹配到 {match['name']} bbox={match['bbox']}")


def _replay_dir():
    """命令行里找 --replay 的目录参数，没有返回 None"""
    if '--replay' in sys.argv:
        i = sys.argv.index('--replay')
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def open_source():
    """返回一个逐帧产出 BGR 图的生成器；支持实时摄像头与回放目录"""
    replay = _replay_dir()
    if replay:
        paths = sorted(glob.glob(os.path.join(replay, '*.jpg')))
        assert paths, f'回放目录无 jpg：{replay}'
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
        result = identify_crop(model, crop)
        print(f'[DEBUG] classify → {result}')
        return result

    def locate_fn(ref, new):
        bboxes = find_change_regions(ref, new)
        print(f'[DEBUG] 变化区域: {len(bboxes)} 个, boxes={bboxes}')
        regions = classify_regions(ref, new, bboxes, classify_fn)
        for r in regions:
            print(f'[DEBUG] 区域: kind={r.kind} ref={r.ref_ident} new={r.new_ident}')
        mark_package_take_out(regions, inv)
        maybe_save_package_takeout_candidate(ref, new, regions)
        maybe_save_package_candidate(new, regions)
        return regions

    detector = EventDetector(is_moving, locate_fn)
    print('✓ 事件检测器就绪')
    if SHOW:
        print('✓ 显示窗口已开启（窗口内按 q 退出）')

    frames = open_source()
    first = next(frames, None)
    assert first is not None, '无可用帧'

    dets = seed_detections(model, first)
    detector.seed(first, dets)          # 始终恢复位置记忆
    if inv.has_stock():
        # 非首次启动：DB 已有库存（可能含用户手动修正）
        # 只恢复 placed_items 位置记忆，不再写 DB，保护用户数据
        print(f'✓ 检测到已有库存，跳过 DB 播种，仅恢复位置记忆（{len(dets)} 个检测框）')
    else:
        # 首次启动：DB 为空，正常初始化库存
        for d in dets:
            inv.process_event('PUT_IN', {'added': {d['class_id']: 1}})
        print(f'✓ 首次启动播种：{len(dets)} 个物品')
    inv.print_stock()

    frame_count = 0
    prev_state = detector.state
    last_event_text = ''
    region_show = 0          # 剩余显示变化框的帧数
    try:
        for frame in frames:
            frame_count += 1
            events, state = detector.update(frame)
            for event_type, details in events:
                inv.process_event(event_type, details)
                last_event_text = format_event(event_type, details)
                print(f'[事件] {event_type} {details}')
                inv.print_stock()

            # 分析刚结束（SETTLING→STABLE）→ 变化框显示一段时间
            if prev_state == 'SETTLING' and state == 'STABLE':
                region_show = REGION_SHOW_FRAMES
            prev_state = state

            if frame_count % PREVIEW_INTERVAL == 0:
                save_preview(frame, state, frame_count, last_event_text)
                print(f'帧{frame_count:5d} | 状态:{state}')

            if SHOW:
                regions = detector.last_regions if region_show > 0 else None
                vis = draw_overlay(frame, state, frame_count,
                                   last_event_text, regions)
                cv2.imshow('Fridge Monitor', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('\n窗口退出')
                    break
            if region_show > 0:
                region_show -= 1
    except KeyboardInterrupt:
        print('\n用户中断')
    finally:
        if SHOW:
            cv2.destroyAllWindows()
        print('\n====== 最终库存 ======')
        inv.print_stock()
        inv.close()
        model.release()
        print('系统退出')


if __name__ == '__main__':
    main()
