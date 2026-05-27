"""录制摄像头帧序列，用于 main.py --replay 复现事件。

用法：
  python3 debug_record_clip.py --out results/banana_miss --seconds 20 --show
  python3 main.py --replay results/banana_miss --show

说明：
  这个脚本不加载模型，只负责按固定间隔保存 jpg。
  用它记录“前两次香蕉没反应”这类场景，之后可以反复回放调参数。
"""
import argparse
import os
import time

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--fps', type=float, default=10)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    assert cap.isOpened(), f'摄像头打开失败: {args.camera}'

    interval = 1.0 / max(args.fps, 1)
    start = time.time()
    next_save = start
    saved = 0
    try:
        while time.time() - start < args.seconds:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            now = time.time()
            if now >= next_save:
                saved += 1
                path = os.path.join(args.out, f'{saved:05d}.jpg')
                cv2.imwrite(path, frame)
                next_save = now + interval
                print(f'保存 {path}')
            if args.show:
                vis = frame.copy()
                cv2.putText(vis, f'saved {saved}', (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                cv2.imshow('debug_record_clip', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f'录制完成: {args.out}, 共 {saved} 帧')


if __name__ == '__main__':
    main()
