"""
冰箱食材识别与管理系统 - 主程序
"""
import cv2
import time
import os
from rknnlite.api import RKNNLite
from utils import preprocess, postprocess, draw_results
from event_detector import EventDetector
from inventory import InventoryManager

RKNN_MODEL = 'models/fridge_yolo_fp16.rknn'
CAMERA_ID = 0
INFER_EVERY = 3


def main():
    model = RKNNLite()
    model.load_rknn(RKNN_MODEL)
    model.init_runtime()
    print('✓ 模型加载成功')

    detector = EventDetector()
    inv = InventoryManager()
    print('✓ 事件检测器 & 库存管理器就绪')

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    assert cap.isOpened(), '摄像头打开失败'
    print('✓ 摄像头就绪\n')

    os.makedirs('results', exist_ok=True)
    inv.print_stock()

    frame_count = 0
    last_boxes, last_confs, last_class_ids = [], [], []
    last_event_type = 'NONE'
    state = 'IDLE'
    has_motion = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_count += 1

            if frame_count % INFER_EVERY == 0:
                img_input, ratio, pad = preprocess(frame)
                outputs = model.inference(inputs=[img_input])

                if outputs is not None and outputs[0] is not None:
                    last_boxes, last_confs, last_class_ids = postprocess(
                        outputs, ratio, pad, frame.shape
                    )

                current_dets = list(zip(last_class_ids, last_confs, last_boxes)) \
                    if len(last_boxes) > 0 else []

                result, state, has_motion = detector.update(frame, current_dets)

                if result:
                    event_type, details, _ = result
                    last_event_type = event_type
                    if event_type != 'NO_EVENT':
                        inv.process_event(event_type, details)
                        inv.print_stock()
            else:
                result, state, has_motion = detector.update(frame, [])

            # 显示
            display = frame.copy()
            if len(last_boxes) > 0:
                display = draw_results(display, last_boxes, last_confs, last_class_ids)

            state_color = {
                'IDLE': (0,255,0),
                'DOOR_OPEN': (0,165,255),
                'COMPARING': (0,0,255)
            }.get(state, (255,255,255))

            cv2.putText(display, f'State:{state}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            cv2.putText(display, f'Motion:{"YES" if has_motion else "NO"}',
                (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,0,255) if has_motion else (0,255,0), 2)
            cv2.putText(display, f'Targets:{len(last_boxes)}',
                (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(display, f'Event:{last_event_type}',
                (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

            cv2.imshow('Fridge System', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_count % 30 == 0:
                stock = inv.get_current_stock()
                stock_str = ', '.join(
                    [f'{n}x{q}' for n,q,_,_ in stock]
                ) or '空'
                print(f'帧{frame_count:5d} | {state:10s} | '
                      f'运动:{"有" if has_motion else "无"} | '
                      f'目标:{len(last_boxes):2d} | 库存:[{stock_str}]')

    except KeyboardInterrupt:
        print('\n用户中断')

    finally:
        print('\n====== 最终库存 ======')
        inv.print_stock()
        print('====== 事件记录 ======')
        for row in inv.get_recent_events(20):
            print(f'  {row[0]} | {row[1]:20s} | {row[2]}')
        cv2.destroyAllWindows()
        cap.release()
        model.release()
        inv.close()
        print('系统退出')


if __name__ == '__main__':
    main()