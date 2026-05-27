"""Standalone RKNN YOLOv8 model tester for RK3568.

This script tests the model itself only. It does not touch inventory,
events, SQLite, Flask, or the refrigerator main loop.

Examples:
  python3 rknn_model_tester.py --camera 0 --show
  python3 rknn_model_tester.py --image path/to/test.jpg --show
  python3 rknn_model_tester.py --dir path/to/images --save-dir results/model_test
"""
import argparse
import csv
import glob
import shutil
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MODEL = 'models/fridge_yolo_v2.rknn'
IMG_SIZE = 640
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

CLASSES = [
    'apple', 'Onion', 'banana', 'garlic', 'pear',
    'orange', 'Capsicum', 'Beet', 'Tomato', 'Cucumber',
    'carrot', 'Eggplant', 'Cabbage', 'Potato', 'Zucchini',
    'pineapple', 'Garlic', 'Cauliflower', 'Calabash',
]


def class_colors(num_classes):
    """Return stable BGR colors, one per class."""
    rng = np.random.default_rng(42)
    colors = rng.integers(40, 230, size=(num_classes, 3), dtype=np.uint8)
    return [tuple(int(v) for v in color) for color in colors]


COLORS = class_colors(len(CLASSES))


def letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE), color=(114, 114, 114)):
    """Resize with unchanged aspect ratio and pad to new_shape."""
    shape = img.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, (left, top)


def preprocess(img_bgr):
    """BGR -> RGB, letterbox to 640x640, add batch dimension."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lb, ratio, pad = letterbox(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_input = np.expand_dims(img_lb, axis=0)
    return img_input, ratio, pad


def nms(boxes, scores, iou_thresh):
    """Pure numpy NMS for xyxy boxes."""
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        remain = np.where(iou <= iou_thresh)[0]
        order = order[remain + 1]

    return keep


def postprocess(
    outputs,
    ratio,
    pad,
    orig_shape,
    conf_thresh=0.20,
    nms_thresh=0.45,
    max_candidates=300,
    box_format='xyxy',
    box_scale='auto',
):
    """Parse RKNN YOLOv8 output shaped (1, 23, 8400, 1)."""
    if outputs is None or len(outputs) == 0 or outputs[0] is None:
        return [], [], []

    pred = outputs[0]
    if pred.ndim == 4:
        pred = pred[:, :, :, 0]
    if pred.ndim != 3 or pred.shape[1] != 4 + len(CLASSES):
        raise ValueError(
            f'unexpected output shape {pred.shape}, expected (1, 23, 8400)'
        )

    pred = pred[0].T  # (8400, 23)
    raw_boxes = pred[:, :4].astype(np.float32)
    max_coord = float(np.max(raw_boxes)) if raw_boxes.size else 0.0
    if box_scale == 'normalized' or (box_scale == 'auto' and max_coord <= 2.0):
        raw_boxes = raw_boxes * IMG_SIZE

    if box_format == 'xywh':
        cx = raw_boxes[:, 0]
        cy = raw_boxes[:, 1]
        bw = raw_boxes[:, 2]
        bh = raw_boxes[:, 3]
        boxes_xyxy = np.stack(
            [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
            axis=1,
        )
    else:
        boxes_xyxy = raw_boxes

    cls_scores = pred[:, 4:]
    class_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
    confidences = np.max(cls_scores, axis=1).astype(np.float32)

    valid = (confidences >= conf_thresh) & (boxes_xyxy.sum(axis=1) > 0)
    boxes_xyxy = boxes_xyxy[valid]
    confidences = confidences[valid]
    class_ids = class_ids[valid]
    if len(boxes_xyxy) == 0:
        return [], [], []

    x1 = (boxes_xyxy[:, 0] - pad[0]) / ratio
    y1 = (boxes_xyxy[:, 1] - pad[1]) / ratio
    x2 = (boxes_xyxy[:, 2] - pad[0]) / ratio
    y2 = (boxes_xyxy[:, 3] - pad[1]) / ratio

    h, w = orig_shape[:2]
    x1 = np.clip(x1, 0, w)
    y1 = np.clip(y1, 0, h)
    x2 = np.clip(x2, 0, w)
    y2 = np.clip(y2, 0, h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    valid_size = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid_size]
    confidences = confidences[valid_size]
    class_ids = class_ids[valid_size]
    if len(boxes) == 0:
        return [], [], []

    if max_candidates and len(confidences) > max_candidates:
        top_idx = confidences.argsort()[::-1][:max_candidates]
        boxes = boxes[top_idx]
        confidences = confidences[top_idx]
        class_ids = class_ids[top_idx]

    keep_all = []
    for cid in np.unique(class_ids):
        idx = np.where(class_ids == cid)[0]
        keep_local = nms(boxes[idx], confidences[idx], nms_thresh)
        keep_all.extend(idx[i] for i in keep_local)

    keep_all = sorted(keep_all, key=lambda i: float(confidences[i]), reverse=True)
    boxes = np.rint(boxes[keep_all]).astype(np.int32)
    confidences = confidences[keep_all]
    class_ids = class_ids[keep_all]
    return boxes, confidences, class_ids


def draw_results(img, boxes, confs, class_ids, infer_ms=None, title=None, fps=None):
    """Draw boxes, class names, and confidence values on a BGR image."""
    for box, conf, cid in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = COLORS[int(cid)]
        label = f'{CLASSES[int(cid)]} {float(conf):.2f}'

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        y_text = max(y1, th + baseline + 4)
        cv2.rectangle(
            img,
            (x1, y_text - th - baseline - 4),
            (x1 + tw + 4, y_text),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (x1 + 2, y_text - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    overlay = []
    if title:
        overlay.append(title)
    if infer_ms is not None:
        overlay.append(f'{infer_ms:.1f} ms')
    if fps is not None:
        overlay.append(f'{fps:.1f} FPS')
    if overlay:
        text = ' | '.join(overlay)
        cv2.putText(
            img,
            text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return img


def load_model(model_path):
    """Create RKNNLite, load model, and initialize runtime."""
    from rknnlite.api import RKNNLite

    rknn = RKNNLite()
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f'load_rknn failed: {model_path}, ret={ret}')

    ret = rknn.init_runtime()
    if ret != 0:
        rknn.release()
        raise RuntimeError(f'init_runtime failed, ret={ret}')
    return rknn


def infer_one(
    rknn,
    img_bgr,
    conf_thresh,
    nms_thresh,
    max_candidates=300,
    box_format='xyxy',
    box_scale='auto',
):
    img_input, ratio, pad = preprocess(img_bgr)
    t0 = time.perf_counter()
    outputs = rknn.inference(inputs=[img_input])
    infer_ms = (time.perf_counter() - t0) * 1000.0
    boxes, confs, class_ids = postprocess(
        outputs,
        ratio,
        pad,
        img_bgr.shape,
        conf_thresh,
        nms_thresh,
        max_candidates,
        box_format,
        box_scale,
    )
    return boxes, confs, class_ids, infer_ms


def print_results(image_id, boxes, confs, class_ids, infer_ms):
    print(f'\n[{image_id}] infer_ms={infer_ms:.1f}, detections={len(boxes)}')
    if len(boxes) == 0:
        print('  no detections')
        return
    for box, conf, cid in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        print(
            f'  class={CLASSES[int(cid)]:<12} '
            f'conf={float(conf):.3f} '
            f'bbox=[{x1}, {y1}, {x2}, {y2}] '
            f'infer_ms={infer_ms:.1f}'
        )


def csv_path_from_arg(save_csv, save_dir):
    if not save_csv:
        return None
    if isinstance(save_csv, str):
        return save_csv
    if save_dir:
        return str(Path(save_dir) / 'detections.csv')
    return 'model_test_detections.csv'


def write_csv(csv_path, rows):
    if not csv_path:
        return
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'image', 'class_name', 'confidence',
                'x1', 'y1', 'x2', 'y2', 'infer_ms',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nCSV saved: {out_path}')


def rows_for_detections(image_id, boxes, confs, class_ids, infer_ms):
    rows = []
    for box, conf, cid in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        rows.append({
            'image': image_id,
            'class_name': CLASSES[int(cid)],
            'confidence': f'{float(conf):.6f}',
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'infer_ms': f'{infer_ms:.3f}',
        })
    return rows


def save_outputs(save_dir, image_id, raw_img, vis_img, save_raw=False):
    if not save_dir:
        return
    out_dir = Path(save_dir)
    vis_dir = out_dir / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(image_id)
    vis_path = vis_dir / f'{stem}_vis.jpg'
    cv2.imwrite(str(vis_path), vis_img)

    if save_raw:
        raw_dir = out_dir / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f'{stem}_raw.jpg'
        cv2.imwrite(str(raw_path), raw_img)


def save_current_frame(save_dir, frame_id, raw_img, vis_img):
    if not save_dir:
        print('No --save-dir set, skip saving current frame.')
        return
    out_dir = Path(save_dir)
    raw_dir = out_dir / 'raw'
    vis_dir = out_dir / 'vis'
    raw_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f'frame_{frame_id:06d}_raw.jpg'
    vis_path = vis_dir / f'frame_{frame_id:06d}_vis.jpg'
    cv2.imwrite(str(raw_path), raw_img)
    cv2.imwrite(str(vis_path), vis_img)
    print(f'Saved current frame: {raw_path}, {vis_path}')


class LatestFrameGrabber:
    """Read camera frames in the background and keep only the newest one."""

    def __init__(self, camera_source, width=640, height=480, buffer_size=1):
        self.camera_source = camera_source
        self.cap = open_capture(camera_source, width, height, buffer_size)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        if not self.cap.isOpened():
            raise RuntimeError(f'failed to open camera: {camera_source}')
        print_capture_info(self.cap)

        self.lock = threading.Lock()
        self.latest = None
        self.seq = 0
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return self

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest = frame
                self.seq += 1

    def read_latest(self, last_seq=None, timeout=1.0):
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                if self.latest is not None and self.seq != last_seq:
                    return self.seq, self.latest.copy()
            time.sleep(0.002)
        return last_seq, None

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.cap.release()


class SyncFrameGrabber:
    """Read frames in the main loop; optionally flush stale buffered frames."""

    def __init__(self, camera_source, width=640, height=480, buffer_size=1, flush_frames=0):
        self.cap = open_capture(camera_source, width, height, buffer_size)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        if not self.cap.isOpened():
            raise RuntimeError(f'failed to open camera: {camera_source}')
        print_capture_info(self.cap)
        self.seq = 0
        self.flush_frames = max(0, int(flush_frames))

    def read_latest(self, last_seq=None, timeout=1.0):
        del last_seq, timeout
        for _ in range(self.flush_frames):
            if not self.cap.grab():
                break
        ok, frame = self.cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            return self.seq, None
        self.seq += 1
        return self.seq, frame

    def release(self):
        self.cap.release()


def parse_camera_source(camera):
    if camera is None:
        return None
    text = str(camera).strip()
    if text.lower() == 'auto':
        return 'auto'
    if text.isdigit():
        return int(text)
    return text


def open_capture(source, width=640, height=480, buffer_size=1):
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def print_capture_info(cap):
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_text = ''.join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print(
        f'Capture actual: width={width:.0f}, height={height:.0f}, '
        f'fps={fps:.1f}, fourcc={fourcc_text!r}'
    )


def probe_camera_source(source, width=640, height=480, buffer_size=1):
    cap = open_capture(source, width, height, buffer_size)
    opened = cap.isOpened()
    ok = False
    shape = None
    if opened:
        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None:
                shape = frame.shape
                break
            time.sleep(0.03)
    cap.release()
    return opened, ok, shape


def list_camera_sources(width=640, height=480, buffer_size=1):
    sources = sorted(glob.glob('/dev/video*'))
    if not sources:
        sources = list(range(8))

    print('Camera probe result:')
    usable = []
    for source in sources:
        opened, ok, shape = probe_camera_source(source, width, height, buffer_size)
        status = 'OK' if ok else ('opened-no-frame' if opened else 'cannot-open')
        shape_text = f', shape={shape}' if shape is not None else ''
        print(f'  {source}: {status}{shape_text}')
        if ok:
            usable.append(source)
    if usable:
        print(f'First usable camera: {usable[0]}')
    else:
        print('No usable OpenCV camera source found.')
    return usable


def resolve_camera_source(args):
    source = parse_camera_source(args.camera)
    if source == 'auto':
        usable = list_camera_sources(
            args.camera_width, args.camera_height, args.camera_buffer
        )
        if not usable:
            raise RuntimeError('no usable camera found for --camera auto')
        return usable[0]
    return source


def safe_stem(image_id):
    stem = Path(str(image_id)).stem
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in stem)


def copy_raw_image(src_path, save_dir):
    raw_dir = Path(save_dir) / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst = raw_dir / Path(src_path).name
    if Path(src_path).resolve() != dst.resolve():
        shutil.copy2(src_path, dst)


def true_class_from_name(path):
    lower_name = Path(path).stem.lower()
    matches = []
    for name in CLASSES:
        if name.lower() in lower_name:
            matches.append(name)
    return matches


def update_stats(stats, boxes, confs, class_ids):
    for conf, cid in zip(confs, class_ids):
        name = CLASSES[int(cid)]
        stats[name]['count'] += 1
        stats[name]['conf_sum'] += float(conf)


def print_batch_summary(stats, total_images, hit_total, labeled_total):
    print('\n====== Batch summary ======')
    print(f'images: {total_images}')
    print('class counts and average confidence:')
    any_detection = False
    for name in CLASSES:
        count = stats[name]['count']
        if count == 0:
            continue
        any_detection = True
        avg_conf = stats[name]['conf_sum'] / count
        print(f'  {name:<12} count={count:<4} avg_conf={avg_conf:.3f}')
    if not any_detection:
        print('  no detections')
    if labeled_total > 0:
        hit_rate = hit_total / labeled_total * 100.0
        print(
            f'rough filename hit rate: {hit_total}/{labeled_total} '
            f'({hit_rate:.1f}%)'
        )


def process_image_path(args, rknn, image_path, csv_rows=None, stats=None):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f'Skip unreadable image: {image_path}')
        return False, False

    boxes, confs, class_ids, infer_ms = infer_one(
        rknn,
        img,
        args.conf,
        args.nms,
        args.max_candidates,
        args.box_format,
        args.box_scale,
    )
    image_id = Path(image_path).name
    print_results(image_id, boxes, confs, class_ids, infer_ms)

    if stats is not None:
        update_stats(stats, boxes, confs, class_ids)

    if csv_rows is not None:
        csv_rows.extend(
            rows_for_detections(image_id, boxes, confs, class_ids, infer_ms)
        )

    true_names = true_class_from_name(image_path)
    detected_names = {CLASSES[int(cid)].lower() for cid in class_ids}
    labeled = len(true_names) > 0
    hit = any(name.lower() in detected_names for name in true_names)
    if labeled:
        print(f'  filename_label={true_names}, rough_hit={hit}')

    vis = draw_results(
        img.copy(), boxes, confs, class_ids, infer_ms, title=image_id
    )
    save_outputs(args.save_dir, image_id, img, vis, save_raw=False)
    if args.save_dir and args.save_raw:
        copy_raw_image(image_path, args.save_dir)

    if args.show:
        cv2.imshow('rknn_model_tester', vis)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt

    return labeled, hit


def run_image(args, rknn):
    csv_rows = []
    process_image_path(args, rknn, args.image, csv_rows=csv_rows)
    write_csv(csv_path_from_arg(args.save_csv, args.save_dir), csv_rows)


def iter_images(folder):
    root = Path(folder)
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def run_dir(args, rknn):
    csv_rows = []
    stats = defaultdict(lambda: {'count': 0, 'conf_sum': 0.0})
    total_images = 0
    labeled_total = 0
    hit_total = 0

    for image_path in iter_images(args.dir):
        total_images += 1
        try:
            labeled, hit = process_image_path(
                args, rknn, image_path, csv_rows=csv_rows, stats=stats
            )
        except KeyboardInterrupt:
            print('\nStopped by user.')
            break
        labeled_total += int(labeled)
        hit_total += int(hit)

    if total_images == 0:
        print(f'No images found in: {args.dir}')
    print_batch_summary(stats, total_images, hit_total, labeled_total)
    write_csv(csv_path_from_arg(args.save_csv, args.save_dir), csv_rows)


def run_camera(args, rknn):
    camera_source = resolve_camera_source(args)
    print(f'Camera source: {camera_source}')
    if args.capture_mode == 'thread':
        grabber = LatestFrameGrabber(
            camera_source,
            width=args.camera_width,
            height=args.camera_height,
            buffer_size=args.camera_buffer,
        ).start()
    else:
        grabber = SyncFrameGrabber(
            camera_source,
            width=args.camera_width,
            height=args.camera_height,
            buffer_size=args.camera_buffer,
            flush_frames=args.flush_frames,
        )

    csv_rows = []
    paused = False
    frame_id = 0
    last_seq = None
    last_raw = None
    last_vis = None
    last_boxes = []
    last_confs = []
    last_class_ids = []
    last_infer_ms = None
    fps = None
    last_tick = time.perf_counter()
    last_frame_sig = None
    same_frame_count = 0

    try:
        while True:
            if not paused:
                last_seq, frame = grabber.read_latest(last_seq)
                if frame is None:
                    print('No fresh camera frame, retrying...')
                    continue

                frame_id += 1
                frame_sig = int(np.mean(frame[::32, ::32]))
                if frame_sig == last_frame_sig:
                    same_frame_count += 1
                else:
                    same_frame_count = 0
                last_frame_sig = frame_sig
                if same_frame_count == 30:
                    print(
                        'Warning: camera frames look unchanged for 30 loops. '
                        'Try --list-cameras or another /dev/videoX node.'
                    )

                should_infer = (frame_id - 1) % max(1, args.infer_every) == 0
                if should_infer:
                    last_boxes, last_confs, last_class_ids, last_infer_ms = infer_one(
                        rknn,
                        frame,
                        args.conf,
                        args.nms,
                        args.max_candidates,
                        args.box_format,
                        args.box_scale,
                    )
                now = time.perf_counter()
                dt = now - last_tick
                if dt > 0:
                    cur_fps = 1.0 / dt
                    fps = cur_fps if fps is None else fps * 0.85 + cur_fps * 0.15
                last_tick = now

                image_id = f'frame_{frame_id:06d}'
                if (
                    should_infer
                    and args.print_every > 0
                    and frame_id % args.print_every == 0
                ):
                    print_results(
                        image_id,
                        last_boxes,
                        last_confs,
                        last_class_ids,
                        last_infer_ms,
                    )
                if args.save_csv and should_infer:
                    csv_rows.extend(
                        rows_for_detections(
                            image_id,
                            last_boxes,
                            last_confs,
                            last_class_ids,
                            last_infer_ms,
                        )
                    )

                last_raw = frame.copy()
                last_vis = draw_results(
                    frame.copy(),
                    last_boxes,
                    last_confs,
                    last_class_ids,
                    last_infer_ms,
                    title=image_id,
                    fps=fps,
                )

            if args.show and last_vis is not None:
                cv2.imshow('rknn_model_tester', last_vis)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key == ord('q'):
                    break
                if key == ord(' '):
                    paused = not paused
                    print('Paused' if paused else 'Resumed')
                if key == ord('s') and last_raw is not None:
                    save_current_frame(args.save_dir, frame_id, last_raw, last_vis)
            else:
                if frame_id >= args.max_frames > 0:
                    break

    finally:
        grabber.release()
        cv2.destroyAllWindows()
        write_csv(csv_path_from_arg(args.save_csv, args.save_dir), csv_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Standalone RKNNLite YOLOv8 model tester.'
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '--camera',
        help='camera id/path/auto, for example 0, /dev/video11, or auto',
    )
    source.add_argument('--image', help='single image path')
    source.add_argument('--dir', help='image folder path')
    source.add_argument(
        '--list-cameras',
        action='store_true',
        help='probe /dev/video* and print OpenCV-readable camera nodes',
    )

    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--conf', type=float, default=0.20)
    parser.add_argument('--nms', type=float, default=0.45)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save-dir', help='save visualized images and raw frames')
    parser.add_argument('--save-raw', action='store_true', help='save raw images')
    parser.add_argument(
        '--print-every',
        type=int,
        default=10,
        help='camera mode only, print detections every N frames. Use 1 for every frame.',
    )
    parser.add_argument('--camera-width', type=int, default=640)
    parser.add_argument('--camera-height', type=int, default=480)
    parser.add_argument(
        '--capture-mode',
        choices=['sync', 'thread'],
        default='sync',
        help='sync is safer on RK3568; thread keeps only newest frame but may be less stable',
    )
    parser.add_argument(
        '--flush-frames',
        type=int,
        default=0,
        help='sync camera mode only, discard this many buffered frames before read',
    )
    parser.add_argument(
        '--camera-buffer',
        type=int,
        default=1,
        help='try to keep camera driver buffer small to reduce latency',
    )
    parser.add_argument(
        '--max-candidates',
        type=int,
        default=300,
        help='max boxes kept before NMS. Use 0 to disable, larger values may slow down.',
    )
    parser.add_argument(
        '--infer-every',
        type=int,
        default=1,
        help='camera mode only, run RKNN every N frames and reuse last boxes between runs',
    )
    parser.add_argument(
        '--box-format',
        choices=['xyxy', 'xywh'],
        default='xyxy',
        help='model box format. Use xywh if boxes are shifted/wrong-sized.',
    )
    parser.add_argument(
        '--box-scale',
        choices=['auto', 'input', 'normalized'],
        default='auto',
        help='box coordinate scale. auto treats <=2.0 as normalized 0..1.',
    )
    parser.add_argument(
        '--save-csv',
        nargs='?',
        const=True,
        default=False,
        help='save CSV. Optional value is CSV path.',
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=0,
        help='camera mode only, 0 means unlimited when --show is off',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f'Model: {args.model}')
    print(f'conf={args.conf:.3f}, nms={args.nms:.3f}')
    print('Use q to quit, s to save current frame, space to pause/resume.')

    if args.list_cameras:
        list_camera_sources(args.camera_width, args.camera_height, args.camera_buffer)
        return

    rknn = load_model(args.model)
    print('RKNNLite model loaded.')
    try:
        if args.camera is not None:
            run_camera(args, rknn)
        elif args.image:
            run_image(args, rknn)
        elif args.dir:
            run_dir(args, rknn)
    finally:
        rknn.release()
        cv2.destroyAllWindows()
        print('RKNNLite released.')


if __name__ == '__main__':
    main()
