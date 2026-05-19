"""
工具函数：预处理 + YOLOv8后处理
"""
import cv2
import numpy as np

CLASSES = ['Apple', 'Avocado', 'Banana', 'Bell pepper', 'Bread', 'Broccoli',
           'Butter', 'Carrot', 'Cheese', 'Chicken', 'Cooking cream', 'Eggs',
           'Garlic', 'Hot Sauce', 'Ketchup', 'Lemon', 'Tomato']

COARSE_MAP = {
    'Apple': '蔬果', 'Avocado': '蔬果', 'Banana': '蔬果',
    'Bell pepper': '蔬果', 'Broccoli': '蔬果', 'Carrot': '蔬果',
    'Garlic': '蔬果', 'Lemon': '蔬果', 'Tomato': '蔬果',
    'Chicken': '生鲜', 'Eggs': '生鲜',
    'Butter': '乳品', 'Cheese': '乳品', 'Cooking cream': '乳品',
    'Bread': '包装食品', 'Hot Sauce': '包装食品', 'Ketchup': '包装食品',
}


def to_coarse(class_id):
    """细类 id → 粗分类名称"""
    return COARSE_MAP[CLASSES[class_id]]


IMG_SIZE = 416
CONF_THRESH = 0.30
NMS_THRESH = 0.45


def letterbox(img, new_shape=(416, 416), color=(114, 114, 114)):
    """保持宽高比的resize+padding"""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def preprocess(img):
    """预处理：BGR→RGB，letterbox"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, ratio, pad = letterbox(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_input = np.expand_dims(img_resized, 0)  # [1,416,416,3]
    return img_input, ratio, pad


def postprocess(outputs, ratio, pad, orig_shape):
    """
    YOLOv8后处理
    输出格式: (1, 21, 3549, 1)
    前4行是 x1,y1,x2,y2 (像素坐标, 416尺度)
    后17行是类别得分
    """
    pred = outputs[0]
    if pred.ndim == 4:
        pred = pred[:, :, :, 0]
    pred = pred[0]     # (21, 3549)
    pred = pred.T      # (3549, 21)

    boxes_xyxy = pred[:, :4]       # 已经是x1y1x2y2
    cls_scores = pred[:, 4:]       # 17类得分

    class_ids = np.argmax(cls_scores, axis=1)
    confidences = np.max(cls_scores, axis=1)

    # 过滤：置信度 + 坐标不能全是0
    valid = (confidences > CONF_THRESH) & (boxes_xyxy.sum(axis=1) > 0)
    boxes_xyxy = boxes_xyxy[valid]
    confidences = confidences[valid]
    class_ids = class_ids[valid]

    if len(boxes_xyxy) == 0:
        return [], [], []

    # 还原到原图尺寸
    x1 = (boxes_xyxy[:, 0] - pad[0]) / ratio
    y1 = (boxes_xyxy[:, 1] - pad[1]) / ratio
    x2 = (boxes_xyxy[:, 2] - pad[0]) / ratio
    y2 = (boxes_xyxy[:, 3] - pad[1]) / ratio

    x1 = np.clip(x1, 0, orig_shape[1]).astype(int)
    y1 = np.clip(y1, 0, orig_shape[0]).astype(int)
    x2 = np.clip(x2, 0, orig_shape[1]).astype(int)
    y2 = np.clip(y2, 0, orig_shape[0]).astype(int)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS：cv2.dnn.NMSBoxes 要求 [x, y, w, h] 格式，需从 xyxy 转换
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(), confidences.tolist(), CONF_THRESH, NMS_THRESH
    )
    if len(indices) == 0:
        return [], [], []

    indices = np.array(indices).flatten()
    return boxes[indices], confidences[indices], class_ids[indices]


def draw_results(img, boxes, confs, class_ids):
    """在图上绘制检测结果"""
    np.random.seed(42)
    colors = np.random.randint(0, 255, (len(CLASSES), 3)).tolist()

    for box, conf, cls_id in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = box
        color = colors[cls_id]
        label = f'{CLASSES[cls_id]} {conf:.2f}'

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 文字背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def identify_crop(model, crop_bgr):
    """对一张裁剪图运行识别，返回置信度最高的检测结果或 None"""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    img_input, ratio, pad = preprocess(crop_bgr)
    outputs = model.inference(inputs=[img_input])
    if outputs is None or outputs[0] is None:
        return None
    boxes, confs, class_ids = postprocess(
        outputs, ratio, pad, crop_bgr.shape)
    if len(boxes) == 0:
        return None
    best = int(np.argmax(confs))
    cid = int(class_ids[best])
    x1, y1, x2, y2 = boxes[best]
    return {
        'class_id': cid,
        'fine': CLASSES[cid],
        'coarse': to_coarse(cid),
        'conf': float(confs[best]),
        'area': float(abs((x2 - x1) * (y2 - y1))),
    }