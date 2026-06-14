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
import json

import cv2
from rknnlite.api import RKNNLite

from utils import (preprocess, postprocess, identify_crop, to_coarse, CLASSES,
                   draw_results)
from motion import is_moving
from change_locator import find_change_regions, classify_regions
from event_detector import EventDetector
from inventory import InventoryManager
from package_ocr import (save_package_candidate,
                         save_package_takeout_candidate)

RKNN_MODEL = 'models/fridge_yolo_opset11_rknn16.rknn'
CAMERA_ID = 0
PREVIEW_PATH = '/tmp/fridge_latest.jpg'
DEBUG_CROP_DIR = 'debug_crops'
PREVIEW_INTERVAL = 30          # 每30帧保存一次预览（约1秒）
REGION_SHOW_FRAMES = 90        # 分析结束后变化框持续显示的帧数
PACKAGE_OCR_CONF_TRIGGER = 0.35  # YOLO 低置信度时尝试 OCR 辅助包装建档

EGG_CLASS_ID = CLASSES.index('egg')
EGG_COUNT_MIN_DELTA = 2
EGG_COUNT_MIN_TOTAL = 4
EGG_COUNT_CONF_THRESH = 0.10
EGG_CROP_EXPAND = 2.0
SHOW_INFER_INTERVAL = 5

SHOW = '--show' in sys.argv    # 是否开实时显示窗口
SHOW_INFER = '--show-infer' in sys.argv

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


def draw_overlay(frame, state, frame_count, last_event='', regions=None,
                 detections=None):
    """在帧上叠加状态/帧号/事件/变化区域，返回新图（不改原图）"""
    img = frame.copy()
    if detections is not None:
        boxes, confs, class_ids = detections
        img = draw_results(img, boxes, confs, class_ids)
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

def count_class_full_frame(model, frame, class_id, conf_thresh=None):
    """Run YOLO on one frame or crop and count one class."""
    img_input, ratio, pad = preprocess(frame)
    outputs = model.inference(inputs=[img_input])
    if outputs is None or outputs[0] is None:
        return 0
    _, _, class_ids = postprocess(
        outputs, ratio, pad, frame.shape, conf_thresh=conf_thresh)
    if len(class_ids) == 0:
        return 0
    return int(sum(int(cid) == class_id for cid in class_ids))


def merge_region_bboxes(regions):
    if not regions:
        return None
    x1 = min(r.bbox[0] for r in regions)
    y1 = min(r.bbox[1] for r in regions)
    x2 = max(r.bbox[0] + r.bbox[2] for r in regions)
    y2 = max(r.bbox[1] + r.bbox[3] for r in regions)
    return x1, y1, x2 - x1, y2 - y1


def expand_bbox(bbox, frame_shape, scale):
    x, y, w, h = bbox
    img_h, img_w = frame_shape[:2]
    cx = x + w / 2
    cy = y + h / 2
    nw = min(img_w, max(w * scale, w + 40))
    nh = min(img_h, max(h * scale, h + 40))
    x1 = max(0, int(round(cx - nw / 2)))
    y1 = max(0, int(round(cy - nh / 2)))
    x2 = min(img_w, int(round(cx + nw / 2)))
    y2 = min(img_h, int(round(cy + nh / 2)))
    return x1, y1, x2 - x1, y2 - y1


def crop_by_bbox(frame, bbox):
    x, y, w, h = bbox
    return frame[y:y + h, x:x + w]


def save_event_crops(ref_frame, new_frame, regions):
    """保存事件最终变化区域的动作前后裁剪图，供离线分析与复盘。"""
    if not regions:
        return None

    stamp = time.strftime('%Y%m%d_%H%M%S')
    idx = getattr(save_event_crops, '_idx', 0) + 1
    save_event_crops._idx = idx
    event_id = f'event_{stamp}_{idx:03d}'
    out_dir = os.path.join(DEBUG_CROP_DIR, event_id)
    os.makedirs(out_dir, exist_ok=True)

    items = []
    for i, region in enumerate(regions, start=1):
        before = crop_by_bbox(ref_frame, region.bbox)
        after = crop_by_bbox(new_frame, region.bbox)
        before_name = f'final_{i:02d}_ref.jpg'
        after_name = f'final_{i:02d}_new.jpg'
        if before.size:
            cv2.imwrite(os.path.join(out_dir, before_name), before)
        if after.size:
            cv2.imwrite(os.path.join(out_dir, after_name), after)
        items.append({
            'index': i,
            'bbox': list(region.bbox),
            'kind': region.kind,
            'before_file': before_name,
            'after_file': after_name,
            'ref_ident': region.ref_ident,
            'new_ident': region.new_ident,
        })

    with open(os.path.join(out_dir, 'metadata.json'), 'w',
              encoding='utf-8') as f:
        json.dump({
            'event_id': event_id,
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'regions': items,
        }, f, ensure_ascii=False, indent=2)
    print(f'[DEBUG_CROP] saved {out_dir}')
    return out_dir


def apply_zoom_crop_egg_delta(model, ref_frame, new_frame, regions):
    """Use zoomed change crop YOLO count delta for dense egg trays."""
    bbox = merge_region_bboxes(regions)
    if bbox is None:
        return regions
    bbox = expand_bbox(bbox, new_frame.shape, EGG_CROP_EXPAND)
    ref_crop = crop_by_bbox(ref_frame, bbox)
    new_crop = crop_by_bbox(new_frame, bbox)
    if ref_crop.size == 0 or new_crop.size == 0:
        return regions

    ref_count = count_class_full_frame(
        model, ref_crop, EGG_CLASS_ID, conf_thresh=EGG_COUNT_CONF_THRESH)
    new_count = count_class_full_frame(
        model, new_crop, EGG_CLASS_ID, conf_thresh=EGG_COUNT_CONF_THRESH)
    delta = new_count - ref_count
    print(f'[EGG] zoom-crop count: {ref_count} -> {new_count}, '
          f'delta={delta}, bbox={bbox}, conf={EGG_COUNT_CONF_THRESH}')
    if (abs(delta) < EGG_COUNT_MIN_DELTA
            or max(ref_count, new_count) < EGG_COUNT_MIN_TOTAL):
        return regions

    from change_locator import ChangedRegion
    ident = {
        'class_id': EGG_CLASS_ID,
        'fine': 'egg',
        'coarse': to_coarse(EGG_CLASS_ID),
        'conf': 1.0,
        'area': float(bbox[2] * bbox[3]),
        'count': abs(delta),
    }
    if delta > 0:
        return [ChangedRegion(bbox, 'APPEAR', None, ident)]
    return [ChangedRegion(bbox, 'DISAPPEAR', ident, None)]


def infer_full_frame(model, frame):
    """Run YOLO on the current frame for --show debug visualization."""
    img_input, ratio, pad = preprocess(frame)
    outputs = model.inference(inputs=[img_input])
    if outputs is None or outputs[0] is None:
        return None
    return postprocess(outputs, ratio, pad, frame.shape)

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
        regions = apply_zoom_crop_egg_delta(model, ref, new, regions)
        for r in regions:
            print(f'[DEBUG] 区域: kind={r.kind} ref={r.ref_ident} new={r.new_ident}')
        mark_package_take_out(regions, inv)
        maybe_save_package_takeout_candidate(ref, new, regions)
        maybe_save_package_candidate(new, regions)
        save_event_crops(ref, new, regions)
        return regions

    detector = EventDetector(is_moving, locate_fn)
    print('✓ 事件检测器就绪')
    if SHOW:
        print('✓ 显示窗口已开启（窗口内按 q 退出）')
        if SHOW_INFER:
            print(f'✓ 调试推理框已开启，每 {SHOW_INFER_INTERVAL} 帧刷新一次')
        else:
            print('提示：默认只显示状态/变化框；需要实时 YOLO 框请加 --show-infer')

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
    show_detections = None
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
                if SHOW_INFER and frame_count % SHOW_INFER_INTERVAL == 0:
                    show_detections = infer_full_frame(model, frame)
                regions = detector.last_regions if region_show > 0 else None
                vis = draw_overlay(frame, state, frame_count,
                                   last_event_text, regions, show_detections)
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
