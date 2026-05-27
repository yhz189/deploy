"""包装文字识别与候选结果持久化。

OCR 是辅助能力：优先尝试 PaddleOCR，其次尝试 pytesseract。两者都不可用时
返回明确的 unavailable 结果，不影响原有 YOLO 食材识别链路。
"""
import json
import os
import re
import time

import cv2

PACKAGE_CROP_PATH = '/tmp/fridge_package_candidate.jpg'
PACKAGE_META_PATH = '/tmp/fridge_package_candidate.json'
PACKAGE_TAKEOUT_REF_PATH = '/tmp/fridge_package_takeout_ref.jpg'
PACKAGE_TAKEOUT_NEW_PATH = '/tmp/fridge_package_takeout_new.jpg'
PACKAGE_TAKEOUT_META_PATH = '/tmp/fridge_package_takeout.json'

_OCR_ENGINE = None
_OCR_ENGINE_NAME = None
_OCR_INIT_ERROR = None


def _init_ocr():
    """懒加载 OCR 引擎，避免没有 OCR 依赖时影响主程序启动。"""
    global _OCR_ENGINE, _OCR_ENGINE_NAME, _OCR_INIT_ERROR
    if _OCR_ENGINE_NAME is not None or _OCR_INIT_ERROR is not None:
        return

    try:
        from paddleocr import PaddleOCR
        try:
            _OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang='ch',
                                    show_log=False)
        except ValueError:
            # 兼容新版 PaddleOCR 参数名变化。
            _OCR_ENGINE = PaddleOCR(use_textline_orientation=True, lang='ch')
        _OCR_ENGINE_NAME = 'paddleocr'
        return
    except Exception as e:
        _OCR_INIT_ERROR = f'paddleocr unavailable: {e}'

    try:
        import pytesseract
        _OCR_ENGINE = pytesseract
        _OCR_ENGINE_NAME = 'pytesseract'
        _OCR_INIT_ERROR = None
    except Exception as e:
        _OCR_INIT_ERROR = f'pytesseract unavailable: {e}'


def _normalize_text(text):
    text = re.sub(r'\s+', '', text or '')
    text = re.sub(r'[^\w\u4e00-\u9fff]+', '', text)
    return text.strip()


def _looks_like_spec(text):
    return bool(re.fullmatch(r'\d+(\.\d+)?(g|kg|ml|mL|l|L|克|千克|毫升|升)?',
                             text))


def _choose_name(items):
    """从 OCR 行结果中挑一个更像商品名的候选。"""
    candidates = []
    for text, conf in items:
        norm = _normalize_text(text)
        if len(norm) < 2 or _looks_like_spec(norm):
            continue
        has_cn = bool(re.search(r'[\u4e00-\u9fff]', norm))
        score = float(conf) + min(len(norm), 8) * 0.02 + (0.1 if has_cn else 0)
        candidates.append((score, norm, float(conf)))
    if not candidates:
        return None, 0.0
    candidates.sort(reverse=True)
    _, name, conf = candidates[0]
    return name, conf


def _flatten_paddle_result(result):
    items = []
    if not result:
        return items
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if not page:
            continue
        for line in page:
            try:
                text, conf = line[1][0], line[1][1]
                items.append((str(text), float(conf)))
            except Exception:
                continue
    return items


def recognize_package_text(crop_bgr):
    """识别包装裁剪图文字，返回结构化结果。"""
    if crop_bgr is None or crop_bgr.size == 0:
        return {'ok': False, 'error': 'empty crop'}

    _init_ocr()
    if _OCR_ENGINE_NAME is None:
        return {'ok': False, 'engine': None, 'error': _OCR_INIT_ERROR}

    try:
        if _OCR_ENGINE_NAME == 'paddleocr':
            result = _OCR_ENGINE.ocr(crop_bgr, cls=True)
            items = _flatten_paddle_result(result)
        else:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            data = _OCR_ENGINE.image_to_data(
                rgb, lang='chi_sim+eng',
                output_type=_OCR_ENGINE.Output.DICT)
            items = []
            for text, conf in zip(data.get('text', []), data.get('conf', [])):
                try:
                    c = float(conf) / 100.0
                except Exception:
                    c = 0.0
                if text and c > 0:
                    items.append((text, c))
    except Exception as e:
        return {'ok': False, 'engine': _OCR_ENGINE_NAME, 'error': str(e)}

    name, conf = _choose_name(items)
    raw_text = [{'text': t, 'confidence': c} for t, c in items]
    if not name:
        return {'ok': False, 'engine': _OCR_ENGINE_NAME,
                'error': 'no usable text', 'raw_text': raw_text}
    return {'ok': True, 'engine': _OCR_ENGINE_NAME, 'name': name,
            'confidence': conf, 'raw_text': raw_text}


def save_package_candidate(crop_bgr, ocr_result, bbox=None):
    """保存最新包装候选，供 Web/App 查询与确认。"""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    os.makedirs(os.path.dirname(PACKAGE_CROP_PATH), exist_ok=True)
    cv2.imwrite(PACKAGE_CROP_PATH, crop_bgr)
    candidate = {
        'ok': bool(ocr_result.get('ok')),
        'name': ocr_result.get('name') or '',
        'confidence': float(ocr_result.get('confidence') or 0.0),
        'engine': ocr_result.get('engine'),
        'raw_text': ocr_result.get('raw_text', []),
        'error': ocr_result.get('error'),
        'bbox': bbox,
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'image_url': '/package/candidate/image',
    }
    with open(PACKAGE_META_PATH, 'w', encoding='utf-8') as f:
        json.dump(candidate, f, ensure_ascii=False, indent=2)
    return candidate


def load_package_candidate():
    if not os.path.exists(PACKAGE_META_PATH):
        return None
    try:
        with open(PACKAGE_META_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def clear_package_candidate():
    for path in (PACKAGE_META_PATH, PACKAGE_CROP_PATH):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def save_package_takeout_candidate(ref_crop, new_crop, bbox=None, reason=''):
    """保存包装取出候选的前后裁剪图，供手机端 OCR 判断。"""
    if ref_crop is None or new_crop is None:
        return None
    if ref_crop.size == 0 or new_crop.size == 0:
        return None
    os.makedirs(os.path.dirname(PACKAGE_TAKEOUT_REF_PATH), exist_ok=True)
    cv2.imwrite(PACKAGE_TAKEOUT_REF_PATH, ref_crop)
    cv2.imwrite(PACKAGE_TAKEOUT_NEW_PATH, new_crop)
    candidate = {
        'ok': False,
        'engine': 'phone_ocr',
        'error': 'pending phone OCR',
        'bbox': bbox,
        'reason': reason,
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'ref_image_url': '/package/takeout/ref_image',
        'new_image_url': '/package/takeout/new_image',
    }
    with open(PACKAGE_TAKEOUT_META_PATH, 'w', encoding='utf-8') as f:
        json.dump(candidate, f, ensure_ascii=False, indent=2)
    return candidate


def load_package_takeout_candidate():
    if not os.path.exists(PACKAGE_TAKEOUT_META_PATH):
        return None
    try:
        with open(PACKAGE_TAKEOUT_META_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def clear_package_takeout_candidate():
    for path in (PACKAGE_TAKEOUT_META_PATH, PACKAGE_TAKEOUT_REF_PATH,
                 PACKAGE_TAKEOUT_NEW_PATH):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
