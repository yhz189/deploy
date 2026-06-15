"""离线评估固定测量区域内的胡萝卜面积与数量等级。"""
import argparse
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from carrot_area import compare_carrot_area, mask_overlay, measure_carrot_area


def _load(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f'无法读取图片: {path}')
    return image


def _without_masks(result):
    data = dict(result)
    data['before'] = {k: v for k, v in result['before'].items() if k != 'mask'}
    data['after'] = {k: v for k, v in result['after'].items() if k != 'mask'}
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', required=True, help='动作前图片')
    parser.add_argument('--after', required=True, help='动作后图片')
    parser.add_argument('--roi', required=True, type=int, nargs=4,
                        metavar=('X', 'Y', 'W', 'H'), help='固定测量区域')
    parser.add_argument('--output-dir', default='carrot_area_output')
    parser.add_argument('--class-name', default='carrot')
    parser.add_argument('--empty', help='同机位空货架校准图')
    parser.add_argument('--full', help='同机位满量胡萝卜校准图')
    parser.add_argument('--change-threshold', type=float, default=0.15)
    args = parser.parse_args()

    if args.class_name.lower() != 'carrot':
        raise ValueError('当前离线验证工具仅允许 carrot 类别门禁')

    before, after = _load(args.before), _load(args.after)
    if before.shape != after.shape:
        raise ValueError('动作前后图片尺寸必须一致')

    empty_area = 0
    full_area = None
    if args.empty:
        empty_area = measure_carrot_area(_load(args.empty), roi=args.roi)['area_px']
    if args.full:
        full_area = measure_carrot_area(_load(args.full), roi=args.roi)['area_px']

    result = compare_carrot_area(
        before, after, args.roi, change_threshold=args.change_threshold,
        empty_area_px=empty_area, full_area_px=full_area)
    output = _without_masks(result)
    output['class_gate'] = 'carrot'
    output['capability'] = 'offline_validation_only'
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
                mask_overlay(before, args.roi, result['before']['mask']))
    cv2.imwrite(os.path.join(args.output_dir, 'after_overlay.jpg'),
                mask_overlay(after, args.roi, result['after']['mask']))
    with open(os.path.join(args.output_dir, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
