"""
库存查询 Web 界面
访问 http://<板子IP>:5000
"""
import os
from flask import Flask, jsonify, render_template_string, send_file, request
from inventory import InventoryManager
from utils import CLASSES
from package_ocr import (
    PACKAGE_CROP_PATH, PACKAGE_TAKEOUT_NEW_PATH, PACKAGE_TAKEOUT_REF_PATH,
    clear_package_candidate, clear_package_takeout_candidate,
    load_package_candidate, load_package_takeout_candidate,
)

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="3">
    <title>冰箱食材管理系统</title>
    <style>
        body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
               max-width: 800px; margin: 24px auto; padding: 0 16px; background:#f5f5f5; color:#333; }
        h1 { text-align: center; font-size: 1.6em; margin: 0.4em 0; }
        h2 { font-size: 1.15em; margin: 0 0 12px 0; }
        .card { background: white; border-radius: 8px; padding: 16px; margin: 14px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        table { width: 100%; border-collapse: collapse; }
        th { background: #4CAF50; color: white; padding: 8px; text-align: left; font-size: 14px; }
        td { padding: 8px; border-bottom: 1px solid #eee; font-size: 14px; word-break: break-word; }
        tr:hover { background: #f9f9f9; }
        .empty { color: #999; text-align: center; padding: 20px; }
        .badge { background:#4CAF50; color:white; padding:2px 8px; border-radius:12px; font-size:12px; white-space:nowrap; }
        .badge-out { background:#f44336; }
        .badge-partial { background:#ff9800; }
        .time { color: #888; font-size: 12px; }
        @media (max-width: 600px) {
            body { margin: 8px auto; padding: 0 8px; }
            h1 { font-size: 1.3em; }
            h2 { font-size: 1.05em; }
            .card { padding: 12px; margin: 10px 0; }
            th, td { padding: 6px 4px; font-size: 13px; }
            .time { font-size: 11px; }
            .badge { font-size: 11px; padding: 2px 6px; }
        }
    </style>
</head>
<body>
    <h1>🧊 冰箱食材管理系统</h1>

    <div class="card">
        <h2>📦 当前库存</h2>
        {% if stock %}
        <table>
            <tr><th>名称</th><th>类型</th><th>数量</th><th>入库时间</th><th>最后更新</th></tr>
            {% for item in stock %}
            <tr>
                <td>{{ item.name }}</td>
                <td>{{ '包装物品' if item.type == 'package' else '生鲜食材' }}</td>
                <td><span class="badge">{{ item.qty }} 个</span></td>
                <td class="time">{{ item.first_in }}</td>
                <td class="time">{{ item.last_update }}</td>
            </tr>
            {% endfor %}
        </table>
        {% else %}
        <p class="empty">冰箱是空的</p>
        {% endif %}
    </div>

    <div class="card">
        <h2>📋 最近事件（最新20条）</h2>
        {% if events %}
        <table>
            <tr><th>时间</th><th>事件</th><th>详情</th></tr>
            {% for time, etype, note in events %}
            <tr>
                <td class="time">{{ time }}</td>
                <td>
                    {% if 'PUT_IN' in etype %}
                    <span class="badge">放入</span>
                    {% elif 'PARTIAL' in etype %}
                    <span class="badge badge-partial">部分取出·估计</span>
                    {% else %}
                    <span class="badge badge-out">取出</span>
                    {% endif %}
                </td>
                <td>{{ note }}</td>
            </tr>
            {% endfor %}
        </table>
        {% else %}
        <p class="empty">暂无事件记录</p>
        {% endif %}
    </div>

    <div class="card">
        <h2>📷 摄像头画面</h2>
        <img id="cam" src="/camera" style="width:100%;border-radius:4px;background:#000;">
        <p class="time" id="cam-ts" style="text-align:center;margin-top:6px;">加载中...</p>
    </div>

    <p style="text-align:center;color:#aaa;font-size:12px;">库存每3秒刷新 · 画面每1秒刷新</p>
    <script>
    function refreshCam() {
        var img = document.getElementById('cam');
        img.src = '/camera?' + Date.now();
        document.getElementById('cam-ts').innerText = new Date().toLocaleTimeString();
    }
    setInterval(refreshCam, 1000);
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    inv = InventoryManager()
    stock = inv.get_current_stock_records()
    events = inv.get_recent_events(20)
    inv.close()
    return render_template_string(HTML, stock=stock, events=events)

@app.route('/camera')
def camera():
    path = '/tmp/fridge_latest.jpg'
    if not os.path.exists(path):
        return '摄像头未启动', 204
    return send_file(path, mimetype='image/jpeg')


@app.route('/api/stock')
def api_stock():
    inv = InventoryManager()
    stock = inv.get_current_stock_records()
    inv.close()
    return jsonify(stock)

@app.route('/api/events')
def api_events():
    inv = InventoryManager()
    events = [{'time':t,'type':e,'note':n}
              for t,e,n in inv.get_recent_events(20)]
    inv.close()
    return jsonify(events)


@app.route('/api/stock/adjust', methods=['POST'])
def api_adjust():
    """用户手动修正库存数量
    POST JSON: {"name": "banana", "qty": 4}
    """
    data = request.get_json(silent=True)
    if not data or 'name' not in data or 'qty' not in data:
        return jsonify({'ok': False, 'error': '缺少 name 或 qty 字段'}), 400
    inv = InventoryManager()
    ok, msg = inv.adjust_quantity(data['name'], data['qty'])
    inv.close()
    return jsonify({'ok': ok, 'msg': msg}), (200 if ok else 404)


@app.route('/api/package/candidate')
def api_package_candidate():
    """返回最近一次疑似包装物品裁剪图，供 App 做本地 OCR。"""
    cand = load_package_candidate()
    if cand is None:
        return jsonify({'ok': False, 'error': '暂无包装候选'}), 404
    return jsonify(cand)


@app.route('/package/candidate/image')
def package_candidate_image():
    """返回最近一次包装变化区域裁剪图。"""
    if not os.path.exists(PACKAGE_CROP_PATH):
        return '暂无包装裁剪图', 204
    return send_file(PACKAGE_CROP_PATH, mimetype='image/jpeg')


@app.route('/api/package/confirm', methods=['POST'])
def api_package_confirm():
    """手机端 OCR 后提交包装物品入库
    POST JSON: {"name": "纯牛奶", "qty": 1}
    """
    data = request.get_json(silent=True)
    if not data or 'name' not in data:
        return jsonify({'ok': False, 'error': '缺少 name 字段'}), 400
    qty = data.get('qty', 1)
    cand = load_package_candidate() or {}
    bbox = data.get('bbox') or cand.get('bbox')
    source = data.get('source') or 'phone_ocr'
    inv = InventoryManager()
    ok, msg = inv.put_package(
        data['name'], qty, source=source, bbox=bbox,
        expire_date=data.get('expire_date'), category=data.get('category'),
        shelf_id=data.get('shelf_id'))
    inv.close()
    if ok:
        clear_package_candidate()
    return jsonify({'ok': ok, 'msg': msg}), (200 if ok else 400)


@app.route('/api/package/adjust', methods=['POST'])
def api_package_adjust():
    """修改包装物品名称和数量
    POST JSON: {"old_name": "纯牛奶", "name": "蒙牛纯牛奶", "qty": 2}
    """
    data = request.get_json(silent=True)
    if not data or 'qty' not in data:
        return jsonify({'ok': False, 'error': '缺少 qty 字段'}), 400
    old_name = data.get('old_name') or data.get('name')
    new_name = data.get('name') or old_name
    inv = InventoryManager()
    ok, msg = inv.adjust_package(
        old_name, new_name, data['qty'],
        expire_date=data.get('expire_date'), category=data.get('category'),
        shelf_id=data.get('shelf_id'))
    inv.close()
    return jsonify({'ok': ok, 'msg': msg}), (200 if ok else 404)


@app.route('/api/cloud/confirm', methods=['POST'])
def api_cloud_confirm():
    """App/云端识别兜底结果写回
    POST JSON: {"name": "牛奶", "qty": 1, "item_type": "package"}
    """
    data = request.get_json(silent=True)
    if not data or 'name' not in data:
        return jsonify({'ok': False, 'error': '缺少 name 字段'}), 400
    item_type = data.get('item_type', 'package')
    qty = max(1, int(data.get('qty', 1)))
    inv = InventoryManager()
    if item_type == 'fresh':
        try:
            cid = next(i for i, name in enumerate(CLASSES)
                       if name == data['name'])
        except StopIteration:
            cid = None
        if cid is not None:
            inv.process_event('PUT_IN', {'added': {cid: qty}})
            ok, msg = True, f'云端识别生鲜入库：{data["name"]} x{qty}'
        else:
            ok, msg = inv.put_package(
                data['name'], qty, source='cloud',
                expire_date=data.get('expire_date'),
                category=data.get('category') or '云端识别',
                shelf_id=data.get('shelf_id'))
    else:
        cand = load_package_candidate() or {}
        bbox = data.get('bbox') or cand.get('bbox')
        ok, msg = inv.put_package(
            data['name'], qty, source='cloud', bbox=bbox,
            expire_date=data.get('expire_date'),
            category=data.get('category') or '包装食品',
            shelf_id=data.get('shelf_id'))
    inv.close()
    if ok:
        clear_package_candidate()
    return jsonify({'ok': ok, 'msg': msg}), (200 if ok else 400)


@app.route('/api/package/takeout_candidate')
def api_package_takeout_candidate():
    """返回最近一次疑似包装取出的前后裁剪图信息，供 App OCR 判断。"""
    cand = load_package_takeout_candidate()
    if cand is None:
        return jsonify({'ok': False, 'error': '暂无包装取出候选'}), 404
    return jsonify(cand)


@app.route('/package/takeout/ref_image')
def package_takeout_ref_image():
    """返回疑似包装取出前的裁剪图。"""
    if not os.path.exists(PACKAGE_TAKEOUT_REF_PATH):
        return '暂无包装取出前裁剪图', 204
    return send_file(PACKAGE_TAKEOUT_REF_PATH, mimetype='image/jpeg')


@app.route('/package/takeout/new_image')
def package_takeout_new_image():
    """返回疑似包装取出后的裁剪图。"""
    if not os.path.exists(PACKAGE_TAKEOUT_NEW_PATH):
        return '暂无包装取出后裁剪图', 204
    return send_file(PACKAGE_TAKEOUT_NEW_PATH, mimetype='image/jpeg')


@app.route('/api/package/takeout', methods=['POST'])
def api_package_takeout():
    """手机端 OCR 判断包装物品被取出后提交出库
    POST JSON: {"name": "火锅底料", "qty": 1}
    """
    data = request.get_json(silent=True)
    if not data or 'name' not in data:
        return jsonify({'ok': False, 'error': '缺少 name 字段'}), 400
    qty = max(1, int(data.get('qty', 1)))
    inv = InventoryManager()
    inv.process_event('PACKAGE_TAKE_OUT',
                      {'removed_package': {data['name']: qty}})
    inv.close()
    clear_package_takeout_candidate()
    return jsonify({'ok': True,
                    'msg': f'包装物品取出：{data["name"]} x{qty}'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
