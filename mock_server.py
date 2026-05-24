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
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, Response, request

app = Flask(__name__)

# ============== 初始基线数据（/dev/reset 会恢复到这里） ==============

_INITIAL_STOCK = [
    {"name": "banana", "qty": 2, "first_in": "2026-05-20 23:21:09",
     "last_update": "2026-05-21 09:30:15"},
    {"name": "apple", "qty": 3, "first_in": "2026-05-21 08:00:00",
     "last_update": "2026-05-21 08:00:00"},
    {"name": "Tomato", "qty": 1, "first_in": "2026-05-21 09:15:30",
     "last_update": "2026-05-21 09:15:30"},
]

_INITIAL_EVENTS = [
    {"time": "2026-05-21 09:30:15", "type": "PARTIAL_TAKE_OUT",
     "note": "部分取出1个banana（估计）"},
    {"time": "2026-05-21 09:15:30", "type": "PUT_IN", "note": "放入1个Tomato"},
    {"time": "2026-05-21 08:00:00", "type": "PUT_IN", "note": "放入3个apple"},
    {"time": "2026-05-20 23:21:09", "type": "PUT_IN", "note": "放入3个banana"},
]

# ============== 运行时数据（/dev 接口会修改） ==============

MOCK_STOCK = copy.deepcopy(_INITIAL_STOCK)
MOCK_EVENTS = copy.deepcopy(_INITIAL_EVENTS)


def _apply_to_stock(etype, food, n):
    """把事件同步到库存，模拟真实后端「事件→库存」的联动"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    item = next((s for s in MOCK_STOCK if s['name'] == food), None)
    if etype == 'PUT_IN':
        if item:
            item['qty'] += n
            item['last_update'] = now
        else:
            MOCK_STOCK.append({'name': food, 'qty': n,
                               'first_in': now, 'last_update': now})
    elif etype in ('TAKE_OUT', 'PARTIAL_TAKE_OUT'):
        if item:
            item['qty'] -= n
            item['last_update'] = now
            if item['qty'] <= 0:
                MOCK_STOCK.remove(item)   # 取空则从库存移除


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


@app.route('/api/stock/adjust', methods=['POST'])
def api_adjust():
    """用户手动修正库存数量（与真实板子接口保持一致）
    POST JSON: {"name": "banana", "qty": 4}
    """
    data = request.get_json(silent=True)
    if not data or 'name' not in data or 'qty' not in data:
        return jsonify({'ok': False, 'error': '缺少 name 或 qty 字段'}), 400
    name = data['name']
    new_qty = max(0, int(data['qty']))
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    item = next((s for s in MOCK_STOCK if s['name'] == name), None)
    if item is None:
        return jsonify({'ok': False, 'error': f'{name} 不在库存中'}), 404
    old_qty = item['qty']
    item['qty'] = new_qty
    item['last_update'] = now
    if new_qty == 0:
        MOCK_STOCK.remove(item)
    MOCK_EVENTS.insert(0, {
        "time": now,
        "type": "MANUAL_ADJUST",
        "note": f'用户手动修正：{name} {old_qty}→{new_qty}'
    })
    return jsonify({'ok': True, 'msg': f'{name} {old_qty}→{new_qty}'})


@app.route('/dev/reset')
def dev_reset():
    """开发用：库存和事件都恢复到初始基线"""
    global MOCK_STOCK, MOCK_EVENTS
    MOCK_STOCK = copy.deepcopy(_INITIAL_STOCK)
    MOCK_EVENTS = copy.deepcopy(_INITIAL_EVENTS)
    return jsonify({"ok": True})


@app.route('/')
def index():
    return ('<h2>冰箱 Mock 后端运行中</h2>'
            '<ul>'
            '<li><a href="/api/stock">/api/stock</a> — 当前库存</li>'
            '<li><a href="/api/events">/api/events</a> — 事件列表</li>'
            '<li><a href="/camera">/camera</a> — 占位摄像头</li>'
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
