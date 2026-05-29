"""
单张图片推理
- Ubuntu环境：使用 RKNN（模拟器）验证流程
- 开发板环境：使用 RKNNLite（NPU真实推理）
"""
import sys
import os
import time
import cv2
import numpy as np
from utils import preprocess, postprocess, draw_results, CLASSES

# ============ 环境自动切换 ============
USE_LITE = True   # 改成True # 在板子上运行前改成 True

if USE_LITE:
    from rknnlite.api import RKNNLite

    print('运行环境：开发板 (RKNNLite)')
else:
    from rknn.api import RKNN

    print('运行环境：Ubuntu (RKNN 模拟器)')

# ============ 配置 ============
RKNN_MODEL = 'models/fridge_yolo_opset11_rknn16.rknn'
IMG_PATH = 'images/test.jpg'
OUTPUT_PATH = 'results/test_result.jpg'


def init_model():
    """初始化模型"""
    if USE_LITE:
        model = RKNNLite()
        ret = model.load_rknn(RKNN_MODEL)
        assert ret == 0, '加载rknn失败'
        ret = model.init_runtime()
        assert ret == 0, 'init_runtime失败'
    else:
        model = RKNN(verbose=False)
        ret = model.load_rknn(RKNN_MODEL)
        assert ret == 0, '加载rknn失败'
        ret = model.init_runtime()  # 模拟器
        assert ret == 0, 'init_runtime失败'
    return model


def main():
    print(f'加载模型: {RKNN_MODEL}')
    model = init_model()

    # 读图
    img_orig = cv2.imread(IMG_PATH)

    print(f'图片尺寸: {img_orig.shape}')

    # 预处理
    img_input, ratio, pad = preprocess(img_orig)

    # 推理
    t0 = time.time()
    outputs = model.inference(inputs=[img_input])
    t1 = time.time()
    print(f'推理耗时: {(t1 - t0) * 1000:.1f}ms')
    print(f'输出shape: {[o.shape for o in outputs]}')

    # 后处理
    boxes, confs, class_ids = postprocess(outputs, ratio, pad, img_orig.shape)
    print(f'检测到 {len(boxes)} 个目标')
    for box, conf, cls_id in zip(boxes, confs, class_ids):
        print(f'  {CLASSES[cls_id]}: {conf:.2f} @ {box.tolist()}')

    # 画框保存
    if len(boxes) > 0:
        result_img = draw_results(img_orig.copy(), boxes, confs, class_ids)
    else:
        result_img = img_orig

    os.makedirs('results', exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, result_img)
    print(f'结果保存: {OUTPUT_PATH}')

    model.release()


if __name__ == '__main__':
    main()
