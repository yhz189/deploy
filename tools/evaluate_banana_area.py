"""离线评估香蕉动作前后面积与数量等级。"""
import argparse
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from banana_area import compare_banana_area, mask_overlay, measure_banana_area


def _load(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f'无法读取图片: {path}')
    return image


def _json_result(result):
    data = dict(result)
    data['before'] = {k: v for k, v in result['before'].items()
                      if k != 'mask'}
    data['after'] = {k: v for k, v in result['after'].items()
                     if k != 'mask'}
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', required=True, help='动作前裁剪图')
    parser.add_argument('--after', required=True, help='动作后裁剪图')
    parser.add_argument('--output-dir', default='banana_area_output')
    parser.add_argument('--class-name', default='banana',
                        help='模拟 YOLO 类别门禁，仅支持 banana')
    parser.add_argument('--empty', help='同尺寸空场景校准图')
    parser.add_argument('--full', help='同尺寸满量香蕉校准图')
    parser.add_argument('--change-threshold', type=float, default=0.15)
    args = parser.parse_args()

    if args.class_name.lower() != 'banana':
        raise ValueError('当前离线验证工具仅支持 banana 类别门禁')

    before = _load(args.before)
    after = _load(args.after)
    if before.shape != after.shape:
        raise ValueError('动作前后裁剪图尺寸必须一致')

    empty_area = 0
    full_area = None
    if args.empty:
        empty = _load(args.empty)
        if empty.shape != before.shape:
            raise ValueError('空场景校准图尺寸必须与事件裁剪图一致')
        empty_area = measure_banana_area(empty)['area_px']
    if args.full:
        full = _load(args.full)
        if full.shape != before.shape:
            raise ValueError('满量校准图尺寸必须与事件裁剪图一致')
        full_area = measure_banana_area(full)['area_px']

    result = compare_banana_area(
        before, after, change_threshold=args.change_threshold,
        empty_area_px=empty_area, full_area_px=full_area)
    output = _json_result(result)
    output['class_gate'] = args.class_name.lower()
    output['calibration'] = {
        'empty_area_px': empty_area,
        'full_area_px': full_area,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.output_dir, 'before_mask.png'),
                result['before']['mask'])
    cv2.imwrite(os.path.join(args.output_dir, 'after_mask.png'),
                result['after']['mask'])
    cv2.imwrite(os.path.join(args.output_dir, 'before_overlay.jpg'),
                mask_overlay(before, result['before']['mask']))
    cv2.imwrite(os.path.join(args.output_dir, 'after_overlay.jpg'),
                mask_overlay(after, result['after']['mask']))
    with open(os.path.join(args.output_dir, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
