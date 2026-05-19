"""
库存查询 Web 界面
访问 http://<板子IP>:5000
"""
from flask import Flask, jsonify, render_template_string
from inventory import InventoryManager

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="3">
    <title>冰箱食材管理系统</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 40px auto; padding: 0 20px; background:#f5f5f5; }
        h1 { color: #333; text-align: center; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; }
        th { background: #4CAF50; color: white; padding: 10px; text-align: left; }
        td { padding: 8px 10px; border-bottom: 1px solid #eee; }
        tr:hover { background: #f9f9f9; }
        .empty { color: #999; text-align: center; padding: 20px; }
        .badge { background:#4CAF50; color:white; padding:2px 8px; border-radius:12px; font-size:12px; }
        .badge-out { background:#f44336; }
        .badge-partial { background:#ff9800; }
        .time { color: #888; font-size: 12px; }
    </style>
</head>
<body>
    <h1>🧊 冰箱食材管理系统</h1>

    <div class="card">
        <h2>📦 当前库存</h2>
        {% if stock %}
        <table>
            <tr><th>食材</th><th>数量</th><th>入库时间</th><th>最后更新</th></tr>
            {% for name, qty, first_in, last_update in stock %}
            <tr>
                <td>{{ name }}</td>
                <td><span class="badge">{{ qty }} 个</span></td>
                <td class="time">{{ first_in }}</td>
                <td class="time">{{ last_update }}</td>
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

    <p style="text-align:center;color:#aaa;font-size:12px;">每3秒自动刷新</p>
</body>
</html>
'''

@app.route('/')
def index():
    inv = InventoryManager()
    stock = inv.get_current_stock()
    events = inv.get_recent_events(20)
    inv.close()
    return render_template_string(HTML, stock=stock, events=events)

@app.route('/api/stock')
def api_stock():
    inv = InventoryManager()
    stock = [{'name':n,'qty':q,'first_in':f,'last_update':l}
             for n,q,f,l in inv.get_current_stock()]
    inv.close()
    return jsonify(stock)

@app.route('/api/events')
def api_events():
    inv = InventoryManager()
    events = [{'time':t,'type':e,'note':n}
              for t,e,n in inv.get_recent_events(20)]
    inv.close()
    return jsonify(events)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)