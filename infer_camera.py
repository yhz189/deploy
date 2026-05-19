"""
冰箱食材识别 + 事件判断 主程序
"""
import cv2
import numpy as np
import time
import os
from rknnlite.api import RKNNLite
from utils import preprocess, postprocess, draw_results, CLASSES, IMG_SIZE
from event_detector import EventDetector

RKNN_MODEL = 'models/fridge_yolo_fp16.rknn'
CAMERA_ID = 0
SAVE_RESULTS = True   # 没显示器时保存图片


def main():
    # 初始化模型
    model = RKNNLite()
    model.load_rknn(RKNN_MODEL)
    model.init_runtime()
    print('✓ 模型加载成功')

    # 初始化事件检测器
    detector = EventDetector()

    # 初始化摄像头
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    assert cap.isOpened(), '摄像头打开失败'
    print('✓ 摄像头就绪')

    os.makedirs('results', exist_ok=True)
    frame_count = 0
    last_boxes, last_confs, last_cls = [], [], []
    last_event = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 每帧都推理（137ms→约7FPS，冰箱场景够用）
        img_input, ratio, pad = preprocess(frame)
        outputs = model.inference(inputs=[img_input])
        # 加这个保护
        if outputs is None or outputs[0] is None:
            print(f'帧{frame_count}: 推理返回None，跳过')
            continue

        boxes, confs, class_ids = postprocess(outputs, ratio, pad, frame.shape)
        boxes, confs, class_ids = postprocess(
            outputs, ratio, pad, frame.shape
        )

        # 打包当前检测结果
        current_dets = list(zip(class_ids, confs, boxes)) if len(boxes) > 0 else []

        # 事件判断
        event, state, has_hand = detector.update(frame, current_dets)
        if event:
            last_event = event

        # 可视化
        display = frame.copy()
        if len(boxes) > 0:
            display = draw_results(display, boxes, confs, class_ids)

        # 状态信息叠加
        state_color = {
            'IDLE': (0, 255, 0),
            'DOOR_OPEN': (0, 165, 255),
            'COMPARING': (0, 0, 255)
        }.get(state, (255, 255, 255))

        cv2.putText(display, f'State: {state}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        cv2.putText(display, f'Hand: {"YES" if has_hand else "NO"}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if has_hand else (0, 255, 0), 2)
        cv2.putText(display, f'Targets: {len(boxes)}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if last_event:
            ev_type = last_event[0]
            cv2.putText(display, f'Last Event: {ev_type}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        if SAVE_RESULTS and frame_count % 15 == 0:
            cv2.imwrite(f'results/frame_{frame_count:05d}.jpg', display)

        # 有显示器就打开这两行
        # cv2.imshow('Fridge', display)
        # if cv2.waitKey(1) & 0xFF == ord('q'): break

        # 打印状态
        if frame_count % 30 == 0:
            print(f'帧{frame_count} | 状态:{state} | 目标:{len(boxes)} | 有手:{has_hand}')

    cap.release()
    model.release()


if __name__ == '__main__':
    main()