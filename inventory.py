"""
库存管理模块 - SQLite
"""
import sqlite3
import time
from utils import CLASSES

DB_PATH = 'inventory.db'


class InventoryManager:

    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                quantity INTEGER DEFAULT 1,
                first_in TEXT,
                last_update TEXT,
                status TEXT DEFAULT 'in'
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time TEXT,
                event_type TEXT,
                class_id INTEGER,
                class_name TEXT,
                quantity_change INTEGER,
                note TEXT
            );
        ''')
        self.conn.commit()

    def _now(self):
        return time.strftime('%Y-%m-%d %H:%M:%S')

    def process_event(self, event_type, details):
        """根据事件类型更新库存"""
        if event_type == 'NO_EVENT':
            return

        now = self._now()

        if event_type == 'PUT_IN':
            for cls_id, qty in details.get('added', {}).items():
                self._put_in(cls_id, qty, now)

        elif event_type in ('TAKE_OUT', 'PARTIAL_TAKE_OUT'):
            for cls_id, qty in details.get('removed', {}).items():
                self._take_out(cls_id, qty, now, event_type)

        elif event_type == 'EXCHANGE':
            for cls_id, qty in details.get('added', {}).items():
                self._put_in(cls_id, qty, now)
            for cls_id, qty in details.get('removed', {}).items():
                self._take_out(cls_id, qty, now, 'TAKE_OUT')

    def _put_in(self, cls_id, qty, now):
        name = CLASSES[cls_id]
        # 检查是否已有该食材在库
        row = self.conn.execute(
            "SELECT id, quantity FROM inventory WHERE class_id=? AND status='in'",
            (cls_id,)
        ).fetchone()

        if row:
            # 已有则数量叠加
            self.conn.execute(
                "UPDATE inventory SET quantity=?, last_update=? WHERE id=?",
                (row[1] + qty, now, row[0])
            )
        else:
            # 新增记录
            self.conn.execute(
                "INSERT INTO inventory (class_id,class_name,quantity,first_in,last_update,status) VALUES (?,?,?,?,?,'in')",
                (cls_id, name, qty, now, now)
            )

        self.conn.execute(
            "INSERT INTO events (event_time,event_type,class_id,class_name,quantity_change,note) VALUES (?,?,?,?,?,?)",
            (now, 'PUT_IN', cls_id, name, qty, f'放入{qty}个{name}')
        )
        self.conn.commit()
        print(f'[库存] 放入: {name} x{qty}')

    def _take_out(self, cls_id, qty, now, event_type):
        name = CLASSES[cls_id]
        row = self.conn.execute(
            "SELECT id, quantity FROM inventory WHERE class_id=? AND status='in'",
            (cls_id,)
        ).fetchone()

        if row:
            new_qty = max(0, row[1] - qty)
            status = 'out' if new_qty == 0 else 'in'
            self.conn.execute(
                "UPDATE inventory SET quantity=?, status=?, last_update=? WHERE id=?",
                (new_qty, status, now, row[0])
            )
            print(f'[库存] 取出: {name} x{qty}，剩余: {new_qty}')
        else:
            print(f'[库存] 警告: 取出{name}但库存无记录，跳过')

        note = f'{"部分取出" if event_type == "PARTIAL_TAKE_OUT" else "取出"}{qty}个{name}'
        self.conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
            (None, now, event_type, cls_id, name, -qty, note)
        )
        self.conn.commit()
        print(f'[库存] 取出: {name} x{qty}')

    def get_current_stock(self):
        """获取当前在库食材"""
        rows = self.conn.execute(
            "SELECT class_name, quantity, first_in, last_update FROM inventory WHERE status='in' AND quantity>0 ORDER BY last_update DESC"
        ).fetchall()
        return rows

    def get_recent_events(self, limit=10):
        """获取最近事件记录"""
        rows = self.conn.execute(
            "SELECT event_time, event_type, note FROM events ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return rows

    def print_stock(self):
        """打印当前库存"""
        stock = self.get_current_stock()
        print('\n====== 当前库存 ======')
        if not stock:
            print('  （空）')
        for name, qty, first_in, last_update in stock:
            print(f'  {name}: {qty}个 | 入库:{first_in} | 更新:{last_update}')
        print('======================\n')

    def close(self):
        self.conn.close()