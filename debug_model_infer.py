"""独立测试 RKNN 模型识别效果。

用法：
  python3 debug_model_infer.py --image images/test.jpg --show
  python3 debug_model_infer.py --image /tmp/tissue.jpg --crop 100,80,220,160
  python3 debug_model_infer.py --camera 0 --show --save-dir results/debug_model

说明：
  这个脚本只测试 YOLO/RKNN 模型本身，不经过事件状态机和库存逻辑。
  适合排查“黄色纸巾为什么被识别成 banana”这类模型/阈值问题。
"""
import argparse
import os
import time

import cv2
from rknnlite.api import RKNNLite

from utils import CLASSES, draw_results, postprocess, preprocess


DEFAULT_MODEL = 'models/fridge_yolo_v2.rknn'


def _parse_crop(text):
    if not text:
        return None
    vals = [int(v.strip()) for v in text.split(',')]
    if len(vals) != 4:
        raise ValueError('--crop 需要格式 x,y,w,h')
    return tuple(vals)


def _load_model(path):
    model = RKNNLite()
    ret = model.load_rknn(path)
    assert ret == 0, f'加载模型失败: {path}'
    ret = model.init_runtime()
    assert ret == 0, 'init_runtime 失败'
    return model


def _infer(model, frame):
    img_input, ratio, pad = preprocess(frame)
    t0 = time.time()
    outputs = model.inference(inputs=[img_input])
    dt = (time.time() - t0) * 1000
    boxes, confs, class_ids = postprocess(outputs, ratio, pad, frame.shape)
    return boxes, confs, class_ids, dt


def _print_result(boxes, confs, class_ids, dt):
    print(f'推理耗时: {dt:.1f} ms | 检测数: {len(boxes)}')
    for box, conf, cid in zip(boxes, confs, class_ids):
        name = CLASSES[int(cid)]
        print(f'  {name:<12} conf={float(conf):.3f} box={box.tolist()}')


def _run_image(args, model):
    img = cv2.imread(args.image)
    assert img is not None, f'读图失败: {args.image}'
    crop = _parse_crop(args.crop)
    if crop:
        x, y, w, h = crop
        img = img[y:y + h, x:x + w]
        assert img.size > 0, f'裁剪区域为空: {crop}'
        print(f'使用裁剪区域: {crop}, crop_shape={img.shape}')
    boxes, confs, class_ids, dt = _infer(model, img)
    _print_result(boxes, confs, class_ids, dt)
    vis = draw_results(img.copy(), boxes, confs, class_ids)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        cv2.imwrite(args.output, vis)
        print(f'结果保存: {args.output}')
    if args.show:
        cv2.imshow('debug_model_infer', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _run_camera(args, model):
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    assert cap.isOpened(), f'摄像头打开失败: {args.camera}'
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    frame_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame_id += 1
            boxes, confs, class_ids, dt = _infer(model, frame)
            if frame_id % args.print_every == 0:
                print(f'\nframe={frame_id}')
                _print_result(boxes, confs, class_ids, dt)
            vis = draw_results(frame.copy(), boxes, confs, class_ids)
            cv2.putText(vis, f'frame {frame_id}', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if args.show:
                cv2.imshow('debug_model_camera', vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('s') and args.save_dir:
                    path = os.path.join(args.save_dir,
                                        f'frame_{frame_id:05d}.jpg')
                    cv2.imwrite(path, frame)
                    print(f'保存原图: {path}')
            elif args.save_dir and frame_id % args.save_every == 0:
                path = os.path.join(args.save_dir, f'frame_{frame_id:05d}.jpg')
                cv2.imwrite(path, frame)
                print(f'保存原图: {path}')
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--image')
    parser.add_argument('--crop', help='x,y,w,h，只测试指定区域')
    parser.add_argument('--camera', type=int)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--output', default='results/debug_model_result.jpg')
    parser.add_argument('--save-dir')
    parser.add_argument('--save-every', type=int, default=30)
    parser.add_argument('--print-every', type=int, default=10)
    args = parser.parse_args()

    if args.image is None and args.camera is None:
        raise SystemExit('必须指定 --image 或 --camera')

    print(f'加载模型: {args.model}')
    model = _load_model(args.model)
    try:
        if args.image:
            _run_image(args, model)
        else:
            _run_camera(args, model)
    finally:
        model.release()


if __name__ == '__main__':
    main()
