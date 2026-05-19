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
