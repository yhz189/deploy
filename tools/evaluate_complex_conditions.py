"""离线评估复杂光照与反光图像增强效果。"""
import argparse
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from image_enhancement import analyze_complex_conditions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='待评估图片')
    parser.add_argument('--output-dir', default='complex_conditions_output')
    args = parser.parse_args()

    original = cv2.imread(args.input)
    if original is None:
        raise FileNotFoundError(f'无法读取图片: {args.input}')

    images, result = analyze_complex_conditions(original)
    os.makedirs(args.output_dir, exist_ok=True)
    outputs = {
        'original.jpg': original,
        'low_light_enhanced.jpg': images['low_light_enhanced'],
        'highlight_mask.png': images['highlight_mask'],
        'highlight_suppressed.jpg': images['highlight_suppressed'],
    }
    for name, image in outputs.items():
        if not cv2.imwrite(os.path.join(args.output_dir, name), image):
            raise OSError(f'无法写入输出图片: {name}')
    result['input'] = os.path.abspath(args.input)
    result['outputs'] = list(outputs)
    with open(os.path.join(args.output_dir, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
