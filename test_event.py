"""
用两张图片模拟 放入/取出 事件，验证对比逻辑是否正确
不需要摄像头
"""
import cv2
import numpy as np
from rknnlite.api import RKNNLite
from utils import preprocess, postprocess, CLASSES
from event_detector import EventDetector

RKNN_MODEL = 'models/fridge_yolo_fp16.rknn'

def infer(model, img_path):
    img = cv2.imread(img_path)
    assert img is not None, f'图片不存在: {img_path}'
    img_input, ratio, pad = preprocess(img)
    outputs = model.inference(inputs=[img_input])
    if outputs is None:
        return [], [], [], img
    boxes, confs, class_ids = postprocess(outputs, ratio, pad, img.shape)
    return boxes, confs, class_ids, img

def main():
    model = RKNNLite()
    model.load_rknn(RKNN_MODEL)
    model.init_runtime()

    detector = EventDetector()

    # 用同一张图模拟"前"和"后"，验证NO_EVENT逻辑
    print('\n=== 测试1：相同场景（应该是NO_EVENT）===')
    boxes1, confs1, cls1, _ = infer(model, 'images/test.jpg')
    boxes2, confs2, cls2, _ = infer(model, 'images/test.jpg')

    dets1 = list(zip(cls1, confs1, boxes1)) if len(boxes1) > 0 else []
    dets2 = list(zip(cls2, confs2, boxes2)) if len(boxes2) > 0 else []

    event_type, details = detector.compare_detections(dets1, dets2)
    print(f'结果: {event_type} {details}')
    print(f'前帧目标: {[CLASSES[c] for c in cls1]}')
    print(f'后帧目标: {[CLASSES[c] for c in cls2]}')

    # 模拟放入：后帧比前帧多一个目标（手动构造）
    print('\n=== 测试2：模拟放入事件 ===')
    dets_before = []   # 空冰箱
    dets_after = [(1, 0.9, [100,100,200,200])]  # 放入一个Avocado
    event_type, details = detector.compare_detections(dets_before, dets_after)
    print(f'结果: {event_type} {details}')
    print(f'预期: PUT_IN')

    # 模拟取出
    print('\n=== 测试3：模拟取出事件 ===')
    dets_before = [(1, 0.9, [100,100,200,200]), (1, 0.85, [300,300,400,400])]
    dets_after = [(1, 0.9, [100,100,200,200])]  # 少了一个
    event_type, details = detector.compare_detections(dets_before, dets_after)
    print(f'结果: {event_type} {details}')
    print(f'预期: PARTIAL_TAKE_OUT 或 TAKE_OUT')

    model.release()
    print('\n测试完成')

if __name__ == '__main__':
    main()