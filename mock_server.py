"""
冰箱后端 API Mock 服务器 — Android App 开发期间用

不需要 RK3568 开发板，在 PC 上跑这个就能完全模拟板子的接口。
让你开发 App 时不用一直开着板子，省 CPU 也省心。

【用法】
1. 启动：           python mock_server.py
2. 查 PC IP：       ipconfig，找 WLAN 的 IPv4，比如 192.168.3.5
3. 手机连同一个 WiFi
4. App 设置里 IP 填 PC 的 IP，端口 5000
5. 如果手机连不上：到「控制面板 → Windows Defender 防火墙 → 允许应用通过」
   放行 python.exe，或第一次访问时弹窗选「允许」

【开发技巧】
- 浏览器访问 http://localhost:5000/dev/add_event/PUT_IN/apple
  手动追加一条事件，并自动同步更新库存（测 App 下拉刷新很方便）
- 加数量：http://localhost:5000/dev/add_event/PUT_IN/apple?n=3
- 浏览器访问 http://localhost:5000/dev/reset 把库存和事件恢复到初始基线
- 直接改下方 _INITIAL_STOCK / _INITIAL_EVENTS 改基线数据，改完重启服务
"""
import copy
import re
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, Response, request

app = Flask(__name__)

# ============== 初始基线数据（/dev/reset 会恢复到这里） ==============

_INITIAL_STOCK = [
    {"name": "banana", "qty": 1, "first_in": "2026-05-20 23:21:09",
     "last_update": "2026-05-21 09:30:15", "type": "fresh",
     "source": "mock", "expire_date": None, "category": "生鲜食材",
     "shelf_id": "single", "days_to_expire": None, "expire_status": "none",
     "amount_mode": "level", "amount_level": "适量", "amount_ratio": 0.52,
     "area_px": 12000, "display_amount": "适量"},
    {"name": "carrot", "qty": 1, "first_in": "2026-05-21 08:30:00",
     "last_update": "2026-05-21 09:20:00", "type": "fresh",
     "source": "mock", "expire_date": None, "category": "生鲜食材",
     "shelf_id": "single", "days_to_expire": None, "expire_status": "none",
     "amount_mode": "level", "amount_level": "少量", "amount_ratio": 0.18,
     "area_px": 4200, "display_amount": "少量"},
    {"name": "apple", "qty": 3, "first_in": "2026-05-21 08:00:00",
     "last_update": "2026-05-21 08:00:00", "type": "fresh",
     "source": "mock", "expire_date": None, "category": "生鲜食材",
     "shelf_id": "single", "days_to_expire": None, "expire_status": "none",
     "amount_mode": "count", "amount_level": None, "amount_ratio": None,
     "area_px": None, "display_amount": "3个"},
    {"name": "Tomato", "qty": 1, "first_in": "2026-05-21 09:15:30",
     "last_update": "2026-05-21 09:15:30", "type": "fresh",
     "source": "mock", "expire_date": None, "category": "生鲜食材",
     "shelf_id": "single", "days_to_expire": None, "expire_status": "none",
     "amount_mode": "count", "amount_level": None, "amount_ratio": None,
     "area_px": None, "display_amount": "1个"},
    {"name": "纯牛奶", "qty": 1, "first_in": "2026-05-21 10:10:00",
     "last_update": "2026-05-21 10:10:00", "type": "package",
     "source": "phone_ocr", "expire_date": "2026-06-01",
     "category": "饮料乳品", "shelf_id": "single",
     "days_to_expire": 7, "expire_status": "normal",
     "amount_mode": "count", "amount_level": None, "amount_ratio": None,
     "area_px": None, "display_amount": "1个"},
]

_INITIAL_EVENTS = [
    {"time": "2026-05-21 09:35:15", "type": "AREA_TAKE_OUT",
     "note": "取出banana，余量：适量→少量，面积变化-3200px"},
    {"time": "2026-05-21 09:30:15", "type": "AREA_PUT_IN",
     "note": "放入banana，余量：少量→适量，面积变化+5400px"},
    {"time": "2026-05-21 09:30:15", "type": "PARTIAL_TAKE_OUT",
     "note": "部分取出1个banana（估计）"},
    {"time": "2026-05-21 09:15:30", "type": "PUT_IN", "note": "放入1个Tomato"},
    {"time": "2026-05-21 08:00:00", "type": "PUT_IN", "note": "放入3个apple"},
    {"time": "2026-05-20 23:21:09", "type": "PUT_IN", "note": "放入3个banana"},
]

# ============== 运行时数据（/dev 接口会修改） ==============

MOCK_STOCK = copy.deepcopy(_INITIAL_STOCK)
MOCK_EVENTS = copy.deepcopy(_INITIAL_EVENTS)
MOCK_PACKAGE_CANDIDATE = {
    "candidate_id": "mock-package-candidate-001",
    "status": "pending_ocr",
    "ok": False,
    "name": "",
    "confidence": 0.0,
    "engine": "phone_ocr",
    "raw_text": [],
    "error": "pending phone OCR",
    "bbox": [120, 90, 260, 180],
    "created_at": "2026-06-14 12:00:00",
    "expires_at": "2099-06-14 12:01:00",
    "image_url": "/package/candidate/image",
}
MOCK_PACKAGE_TAKEOUT_CANDIDATE = {
    "ok": False,
    "engine": "phone_ocr",
    "error": "pending phone OCR",
    "bbox": [120, 90, 260, 180],
    "reason": "mock package text disappeared",
    "time": "2026-05-21 10:20:00",
    "ref_image_url": "/package/takeout/ref_image",
    "new_image_url": "/package/takeout/new_image",
}


def _apply_to_stock(etype, food, n):
    """把事件同步到库存，模拟真实后端「事件→库存」的联动"""
    if food in ('banana', 'carrot'):
        return
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    item = next((s for s in MOCK_STOCK if s['name'] == food), None)
    if etype == 'PUT_IN':
        if item:
            item['qty'] += n
            item['last_update'] = now
        else:
            MOCK_STOCK.append({'name': food, 'qty': n,
                               'first_in': now, 'last_update': now,
                               'type': 'fresh', 'source': 'mock',
                               'expire_date': None, 'category': '生鲜食材',
                               'shelf_id': 'single',
                               'days_to_expire': None,
                               'expire_status': 'none',
                               'amount_mode': 'count',
                               'amount_level': None,
                               'amount_ratio': None, 'area_px': None,
                               'display_amount': f'{n}个'})
    elif etype in ('TAKE_OUT', 'PARTIAL_TAKE_OUT'):
        if item:
            item['qty'] -= n
            item['last_update'] = now
            if item['qty'] <= 0:
                MOCK_STOCK.remove(item)   # 取空则从库存移除


def _valid_package_text(text):
    raw = (text or '').strip()
    compact = re.sub(r'[^\w\u4e00-\u9fff]+', '', raw)
    if not compact or '\ufffd' in raw or len(compact) < 2:
        return False
    if re.fullmatch(r'\d+(\.\d+)?(g|kg|ml|l|克|千克|毫升|升)?',
                    compact, re.IGNORECASE):
        return False
    return bool(re.search(r'[A-Za-z\u4e00-\u9fff]', compact))


# ============== 路由 ==============

@app.route('/api/stock')
def api_stock():
    return jsonify(MOCK_STOCK)


@app.route('/api/events')
def api_events():
    return jsonify(MOCK_EVENTS[:20])


@app.route('/camera')
def camera():
    """返回一张占位摄像头画面，含时间戳让画面看起来在更新"""
    img = np.full((480, 640, 3), 230, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (600, 440), (200, 200, 200), 2)
    cv2.putText(img, 'MOCK CAMERA', (140, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (60, 120, 60), 3)
    cv2.putText(img, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                (110, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
    cv2.putText(img, 'State: STABLE', (210, 335),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)
    ok, buf = cv2.imencode('.jpg', img)
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/api/package/candidate')
def api_package_candidate():
    """模拟疑似包装物品裁剪图，App 端拿图后做 OCR。"""
    if MOCK_PACKAGE_CANDIDATE is None:
        return jsonify({'ok': False, 'error': '暂无包装候选'}), 404
    return jsonify(MOCK_PACKAGE_CANDIDATE)


@app.route('/api/package/candidate', methods=['DELETE'])
def api_delete_package_candidate():
    global MOCK_PACKAGE_CANDIDATE
    if MOCK_PACKAGE_CANDIDATE is None:
        return jsonify({'ok': False, 'error': '暂无包装候选'}), 404
    if request.args.get('candidate_id') != MOCK_PACKAGE_CANDIDATE['candidate_id']:
        return jsonify({'ok': False, 'error': 'candidate_id 不匹配'}), 409
    MOCK_PACKAGE_CANDIDATE = None
    return jsonify({'ok': True, 'msg': '候选已丢弃'})


@app.route('/package/candidate/image')
def package_candidate_image():
    """返回一张模拟包装裁剪图。"""
    if MOCK_PACKAGE_CANDIDATE is None:
        return '', 204
    img = np.full((260, 420, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (25, 30), (395, 230), (40, 120, 220), 3)
    cv2.putText(img, 'PACKAGE OCR', (70, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
    cv2.putText(img, 'Milk 250ml', (80, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3)
    cv2.putText(img, 'name: pure milk', (85, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2)
    ok, buf = cv2.imencode('.jpg', img)
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/api/package/takeout_candidate')
def api_package_takeout_candidate():
    """模拟疑似包装取出的前后裁剪图，App 端拿图后做 OCR。"""
    return jsonify(MOCK_PACKAGE_TAKEOUT_CANDIDATE)


@app.route('/package/takeout/ref_image')
def package_takeout_ref_image():
    """模拟取出前：裁剪图里有包装文字。"""
    img = np.full((260, 420, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (25, 30), (395, 230), (45, 80, 210), 3)
    cv2.putText(img, 'HOT POT', (105, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 3)
    cv2.putText(img, 'soup base', (95, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
    cv2.putText(img, 'takeout ref', (105, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2)
    ok, buf = cv2.imencode('.jpg', img)
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/package/takeout/new_image')
def package_takeout_new_image():
    """模拟取出后：裁剪图里只剩空背景。"""
    img = np.full((260, 420, 3), 225, dtype=np.uint8)
    cv2.rectangle(img, (25, 30), (395, 230), (190, 190, 190), 2)
    cv2.putText(img, 'EMPTY AREA', (110, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (130, 130, 130), 2)
    ok, buf = cv2.imencode('.jpg', img)
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/dev/add_event/<etype>/<food>')
def dev_add_event(etype, food):
    """开发用：追加一条事件，并同步更新库存
    etype 取值：PUT_IN / TAKE_OUT / PARTIAL_TAKE_OUT
    数量：URL 加 ?n=3，默认 1。例：/dev/add_event/PUT_IN/apple?n=3
    """
    n = request.args.get('n', 1, type=int)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    note_map = {
        'PUT_IN': f'放入{n}个{food}',
        'TAKE_OUT': f'取出{n}个{food}',
        'PARTIAL_TAKE_OUT': f'部分取出{n}个{food}（估计）',
    }
    MOCK_EVENTS.insert(0, {
        "time": now,
        "type": etype,
        "note": note_map.get(etype, f'{etype} {food}')
    })
    _apply_to_stock(etype, food, n)        # 关键：事件联动库存
    return jsonify({"ok": True,
                    "events_count": len(MOCK_EVENTS),
                    "stock": MOCK_STOCK})


@app.route('/dev/set_level/<food>/<level>')
def dev_set_level(food, level):
    """开发用：设置香蕉/胡萝卜等级，模拟面积测量结果。"""
    with app.test_request_context(json={
            'name': food, 'level': level,
            'ratio': request.args.get('ratio', type=float)}):
        return api_adjust_level()


@app.route('/api/stock/adjust', methods=['POST'])
def api_adjust():
    """用户手动修正库存数量（与真实板子接口保持一致）
    POST JSON: {"name": "banana", "qty": 4}
    """
    data = request.get_json(silent=True)
    if not data or 'name' not in data or 'qty' not in data:
        return jsonify({'ok': False, 'error': '缺少 name 或 qty 字段'}), 400
    name = data['name']
    if name in ('banana', 'carrot'):
        return jsonify({'ok': False,
                        'error': f'{name} 使用面积等级，请勿按个数修正'}), 400
    new_qty = max(0, int(data['qty']))
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    item = next((s for s in MOCK_STOCK if s['name'] == name), None)
    if item is None:
        return jsonify({'ok': False, 'error': f'{name} 不在库存中'}), 404
    old_qty = item['qty']
    item['qty'] = new_qty
    item['display_amount'] = f'{new_qty}个'
    item['last_update'] = now
    if new_qty == 0:
        MOCK_STOCK.remove(item)
    MOCK_EVENTS.insert(0, {
        "time": now,
        "type": "MANUAL_ADJUST",
        "note": f'用户手动修正：{name} {old_qty}→{new_qty}'
    })
    return jsonify({'ok': True, 'msg': f'{name} {old_qty}→{new_qty}'})


@app.route('/api/stock/adjust-level', methods=['POST'])
def api_adjust_level():
    data = request.get_json(silent=True)
    if not data or 'name' not in data or 'level' not in data:
        return jsonify({'ok': False, 'error': '缺少 name 或 level 字段'}), 400
    name, level = data['name'], data['level']
    if name not in ('banana', 'carrot'):
        return jsonify({'ok': False, 'error': f'{name} 不是等级型库存'}), 400
    ratios = {'无': 0.0, '少量': 0.18, '适量': 0.50, '大量': 0.82}
    if level not in ratios:
        return jsonify({'ok': False, 'error': f'无效等级: {level}'}), 400
    ratio = data.get('ratio')
    ratio = ratios[level] if ratio is None else float(ratio)
    item = next((s for s in MOCK_STOCK if s['name'] == name), None)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if item is None:
        item = {'name': name, 'qty': 0, 'first_in': now,
                'type': 'fresh', 'expire_date': None,
                'category': '生鲜食材', 'shelf_id': 'single',
                'days_to_expire': None, 'expire_status': 'none'}
        MOCK_STOCK.append(item)
    item.update({
        'qty': 0 if level == '无' else 1,
        'last_update': now, 'source': 'manual_level',
        'amount_mode': 'level', 'amount_level': level,
        'amount_ratio': ratio,
        'area_px': int(data.get('area_px', 0)),
        'display_amount': level,
    })
    return jsonify({'ok': True, 'msg': f'{name}余量等级设置为{level}'})


@app.route('/api/package/confirm', methods=['POST'])
def api_package_confirm():
    """手机端 OCR 后提交包装物品入库
    POST JSON: {"candidate_id": "...", "name": "纯牛奶",
                "confidence": 0.86, "qty": 1}
    """
    data = request.get_json(silent=True)
    global MOCK_PACKAGE_CANDIDATE
    if not data or 'name' not in data or 'candidate_id' not in data:
        return jsonify({'ok': False,
                        'error': '缺少 candidate_id 或 name 字段'}), 400
    if MOCK_PACKAGE_CANDIDATE is None:
        return jsonify({'ok': False, 'error': '候选不存在或已过期'}), 409
    if data['candidate_id'] != MOCK_PACKAGE_CANDIDATE['candidate_id']:
        return jsonify({'ok': False, 'error': 'candidate_id 不匹配'}), 409
    try:
        confidence = float(data.get('confidence') or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.70:
        return jsonify({'ok': False, 'error': 'OCR 置信度过低'}), 409
    if not _valid_package_text(data['name']):
        return jsonify({'ok': False, 'error': 'OCR 文本质量不合格'}), 409
    name = data['name'].strip()
    qty = max(1, int(data.get('qty', 1)))
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    item = next((s for s in MOCK_STOCK
                 if s['name'] == name and s.get('type') == 'package'), None)
    if item:
        item['qty'] += qty
        item['last_update'] = now
        item['source'] = data.get('source', 'phone_ocr')
    else:
        MOCK_STOCK.append({
            "name": name, "qty": qty, "first_in": now, "last_update": now,
            "type": "package", "source": data.get('source', 'phone_ocr'),
            "expire_date": data.get('expire_date'),
            "category": data.get('category', '包装食品'),
            "shelf_id": data.get('shelf_id', 'single'),
            "days_to_expire": None,
            "expire_status": "none",
            "amount_mode": "count", "amount_level": None,
            "amount_ratio": None, "area_px": None,
            "display_amount": f"{qty}个",
        })
    note = f'包装物品入库：{name} x{qty}'
    MOCK_EVENTS.insert(0, {"time": now, "type": "PACKAGE_PUT_IN",
                           "note": note})
    MOCK_PACKAGE_CANDIDATE = None
    return jsonify({'ok': True, 'msg': note, 'stock': MOCK_STOCK})


@app.route('/api/package/adjust', methods=['POST'])
def api_package_adjust():
    """修改包装物品名称和数量
    POST JSON: {"old_name": "纯牛奶", "name": "蒙牛纯牛奶", "qty": 2}
    """
    data = request.get_json(silent=True)
    if not data or 'qty' not in data:
        return jsonify({'ok': False, 'error': '缺少 qty 字段'}), 400
    old_name = (data.get('old_name') or data.get('name') or '').strip()
    new_name = (data.get('name') or old_name).strip()
    qty = max(0, int(data['qty']))
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    item = next((s for s in MOCK_STOCK
                 if s['name'] == old_name and s.get('type') == 'package'), None)
    if item is None:
        return jsonify({'ok': False,
                        'error': f'{old_name} 不在包装物品库存中'}), 404
    old_qty = item['qty']
    item['name'] = new_name
    item['qty'] = qty
    item['display_amount'] = f'{qty}个'
    item['last_update'] = now
    item['source'] = 'manual'
    item['expire_date'] = data.get('expire_date', item.get('expire_date'))
    item['category'] = data.get('category', item.get('category', '包装食品'))
    item['shelf_id'] = data.get('shelf_id', item.get('shelf_id', 'single'))
    if qty == 0:
        MOCK_STOCK.remove(item)
    note = f'用户修正包装物品：{old_name} {old_qty}→{new_name} {qty}'
    MOCK_EVENTS.insert(0, {"time": now, "type": "PACKAGE_MANUAL_ADJUST",
                           "note": note})
    return jsonify({'ok': True, 'msg': note, 'stock': MOCK_STOCK})


@app.route('/api/cloud/confirm', methods=['POST'])
def api_cloud_confirm():
    """模拟云端兜底识别结果写回。"""
    data = request.get_json(silent=True)
    if not data or 'name' not in data:
        return jsonify({'ok': False, 'error': '缺少 name 字段'}), 400
    name = data['name'].strip()
    qty = max(1, int(data.get('qty', 1)))
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    item_type = data.get('item_type', 'package')
    item = next((s for s in MOCK_STOCK
                 if s['name'] == name and s.get('type') == item_type), None)
    if item:
        item['qty'] += qty
        item['last_update'] = now
        item['source'] = 'cloud'
    else:
        MOCK_STOCK.append({
            "name": name, "qty": qty, "first_in": now, "last_update": now,
            "type": item_type, "source": "cloud",
            "expire_date": data.get('expire_date'),
            "category": data.get('category', '云端识别'),
            "shelf_id": data.get('shelf_id', 'single'),
            "days_to_expire": None,
            "expire_status": "none",
            "amount_mode": "count", "amount_level": None,
            "amount_ratio": None, "area_px": None,
            "display_amount": f"{qty}个",
        })
    note = f'云端识别入库：{name} x{qty}'
    MOCK_EVENTS.insert(0, {"time": now, "type": "CLOUD_PUT_IN",
                           "note": note})
    return jsonify({'ok': True, 'msg': note, 'stock': MOCK_STOCK})


@app.route('/api/package/takeout', methods=['POST'])
def api_package_takeout():
    """手机端 OCR 判断包装物品被取出后提交出库
    POST JSON: {"name": "纯牛奶", "qty": 1}
    """
    data = request.get_json(silent=True)
    if not data or 'name' not in data:
        return jsonify({'ok': False, 'error': '缺少 name 字段'}), 400
    name = data['name'].strip()
    qty = max(1, int(data.get('qty', 1)))
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    item = next((s for s in MOCK_STOCK
                 if s['name'] == name and s.get('type') == 'package'), None)
    if item:
        item['qty'] -= qty
        item['last_update'] = now
        if item['qty'] <= 0:
            MOCK_STOCK.remove(item)
    note = f'取出包装物品{name} x{qty}'
    MOCK_EVENTS.insert(0, {"time": now, "type": "PACKAGE_TAKE_OUT",
                           "note": note})
    return jsonify({'ok': True, 'msg': note, 'stock': MOCK_STOCK})


@app.route('/dev/reset')
def dev_reset():
    """开发用：库存和事件都恢复到初始基线"""
    global MOCK_STOCK, MOCK_EVENTS, MOCK_PACKAGE_CANDIDATE
    MOCK_STOCK = copy.deepcopy(_INITIAL_STOCK)
    MOCK_EVENTS = copy.deepcopy(_INITIAL_EVENTS)
    MOCK_PACKAGE_CANDIDATE = {
        "candidate_id": "mock-package-candidate-001",
        "status": "pending_ocr",
        "ok": False,
        "name": "",
        "confidence": 0.0,
        "engine": "phone_ocr",
        "raw_text": [],
        "error": "pending phone OCR",
        "bbox": [120, 90, 260, 180],
        "created_at": "2026-06-14 12:00:00",
        "expires_at": "2099-06-14 12:01:00",
        "image_url": "/package/candidate/image",
    }
    return jsonify({"ok": True})


@app.route('/')
def index():
    return ('<h2>冰箱 Mock 后端运行中</h2>'
            '<ul>'
            '<li><a href="/api/stock">/api/stock</a> — 当前库存</li>'
            '<li><a href="/api/events">/api/events</a> — 事件列表</li>'
            '<li><a href="/camera">/camera</a> — 占位摄像头</li>'
            '<li><a href="/api/package/candidate">/api/package/candidate</a>'
            ' — 最近包装裁剪图</li>'
            '<li><a href="/package/candidate/image">/package/candidate/image</a>'
            ' — 包装裁剪图</li>'
            '<li><a href="/api/package/takeout_candidate">/api/package/takeout_candidate</a>'
            ' — 包装取出候选</li>'
            '<li><a href="/package/takeout/ref_image">/package/takeout/ref_image</a>'
            ' — 取出前裁剪图</li>'
            '<li><a href="/package/takeout/new_image">/package/takeout/new_image</a>'
            ' — 取出后裁剪图</li>'
            '<li>POST /api/cloud/confirm — 云端兜底识别结果写回</li>'
            '<li><a href="/dev/add_event/PUT_IN/apple">放入1个apple</a>'
            '（同时更新库存与事件）</li>'
            '<li><a href="/dev/add_event/PUT_IN/apple?n=3">放入3个apple</a></li>'
            '<li><a href="/dev/add_event/TAKE_OUT/banana">取出1个banana</a></li>'
            '<li><a href="/dev/reset">重置库存和事件</a></li>'
            '</ul>')


if __name__ == '__main__':
    print('=' * 60)
    print('  冰箱 Mock 后端启动')
    print('  本机:    http://localhost:5000')
    print('  手机:    http://<本机IP>:5000  (cmd 里 ipconfig 查 IP)')
    print('  加事件:  /dev/add_event/PUT_IN/banana   (可加 ?n=3)')
    print('  重置:    /dev/reset')
    print('=' * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
