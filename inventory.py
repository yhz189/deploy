"""
库存管理模块 - SQLite
"""
import sqlite3
import time
from datetime import date, datetime
from utils import CLASSES

DB_PATH = 'inventory.db'
PACKAGE_IOU_THRESH = 0.25
LEVEL_MANAGED_NAMES = {'banana', 'carrot'}
LEVEL_DEFAULT_RATIOS = {'无': 0.0, '少量': 0.18, '适量': 0.50, '大量': 0.82}


def _iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


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
                status TEXT DEFAULT 'in',
                item_type TEXT DEFAULT 'fresh',
                source TEXT DEFAULT 'yolo'
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
        self._ensure_column('inventory', 'item_type',
                            "TEXT DEFAULT 'fresh'")
        self._ensure_column('inventory', 'source',
                            "TEXT DEFAULT 'yolo'")
        self._ensure_column('inventory', 'bbox_x', 'INTEGER')
        self._ensure_column('inventory', 'bbox_y', 'INTEGER')
        self._ensure_column('inventory', 'bbox_w', 'INTEGER')
        self._ensure_column('inventory', 'bbox_h', 'INTEGER')
        self._ensure_column('inventory', 'expire_date', 'TEXT')
        self._ensure_column('inventory', 'category', 'TEXT')
        self._ensure_column('inventory', 'shelf_id',
                            "TEXT DEFAULT 'single'")
        self._ensure_column('inventory', 'amount_mode',
                            "TEXT DEFAULT 'count'")
        self._ensure_column('inventory', 'amount_level', 'TEXT')
        self._ensure_column('inventory', 'amount_ratio', 'REAL')
        self._ensure_column('inventory', 'area_px', 'INTEGER')
        self.conn.execute(
            "UPDATE inventory SET amount_mode='level', "
            "amount_level=CASE "
            "WHEN quantity<=0 THEN '无' "
            "WHEN quantity=1 THEN '少量' "
            "WHEN quantity=2 THEN '适量' "
            "ELSE '大量' END, "
            "amount_ratio=CASE "
            "WHEN quantity<=0 THEN 0.0 "
            "WHEN quantity=1 THEN 0.18 "
            "WHEN quantity=2 THEN 0.50 "
            "ELSE 0.82 END, "
            "quantity=CASE WHEN quantity<=0 THEN 0 ELSE 1 END "
            "WHERE item_type='fresh' AND class_name IN ('banana','carrot') "
            "AND COALESCE(amount_mode,'count')!='level'"
        )
        self._consolidate_level_items()
        self.conn.commit()

    def _consolidate_level_items(self):
        """历史库中每个等级型食材只保留一条记录。"""
        for name in LEVEL_MANAGED_NAMES:
            rows = self.conn.execute(
                "SELECT id FROM inventory WHERE class_name=? "
                "AND item_type='fresh' ORDER BY id DESC", (name,)
            ).fetchall()
            if len(rows) <= 1:
                continue
            keep_id = rows[0][0]
            self.conn.execute(
                "UPDATE inventory SET quantity=0, status='out' "
                "WHERE class_name=? AND item_type='fresh' AND id!=?",
                (name, keep_id)
            )

    def _ensure_column(self, table, column, ddl):
        cols = [row[1] for row in self.conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()]
        if column not in cols:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _now(self):
        return time.strftime('%Y-%m-%d %H:%M:%S')

    def process_event(self, event_type, details):
        """根据事件类型更新库存"""
        if event_type == 'NO_EVENT':
            return

        now = self._now()

        if event_type == 'PUT_IN':
            for cls_id, qty in details.get('added', {}).items():
                if CLASSES[cls_id] in LEVEL_MANAGED_NAMES:
                    continue
                self._put_in(cls_id, qty, now)

        elif event_type in ('TAKE_OUT', 'PARTIAL_TAKE_OUT'):
            for cls_id, qty in details.get('removed', {}).items():
                if CLASSES[cls_id] in LEVEL_MANAGED_NAMES:
                    continue
                self._take_out(cls_id, qty, now, event_type)

        elif event_type == 'PACKAGE_TAKE_OUT':
            for name, qty in details.get('removed_package', {}).items():
                self._take_out_package(name, qty, now)

        elif event_type == 'EXCHANGE':
            for cls_id, qty in details.get('added', {}).items():
                if CLASSES[cls_id] in LEVEL_MANAGED_NAMES:
                    continue
                self._put_in(cls_id, qty, now)
            for cls_id, qty in details.get('removed', {}).items():
                if CLASSES[cls_id] in LEVEL_MANAGED_NAMES:
                    continue
                self._take_out(cls_id, qty, now, 'TAKE_OUT')

    def has_level_item(self, class_name):
        row = self.conn.execute(
            "SELECT 1 FROM inventory WHERE class_name=? AND amount_mode='level' LIMIT 1",
            (class_name,)
        ).fetchone()
        return row is not None

    def set_area_level(self, cls_id, level, ratio, area_px,
                       source='opencv_area', action=None, delta_area_px=None):
        """设置香蕉/胡萝卜的面积等级；等级是绝对状态，不使用增量个数。"""
        name = CLASSES[cls_id]
        if name not in LEVEL_MANAGED_NAMES:
            return False, f'{name} 不是等级型库存'
        if level not in ('无', '少量', '适量', '大量'):
            return False, f'无效等级: {level}'

        now = self._now()
        ratio = min(1.0, max(0.0, float(ratio)))
        area_px = max(0, int(area_px))
        # A confirmed area increase cannot leave the item at "none".
        if action == 'PUT_IN' and level == '无':
            level = '少量'
            ratio = max(ratio, LEVEL_DEFAULT_RATIOS['少量'])
        row = self.conn.execute(
            "SELECT id, amount_level, first_in FROM inventory "
            "WHERE class_id=? AND item_type='fresh' ORDER BY id DESC LIMIT 1",
            (cls_id,)
        ).fetchone()
        quantity = 0 if level == '无' else 1
        status = 'out' if level == '无' else 'in'

        if row:
            rec_id, old_level, first_in = row
            self.conn.execute(
                "UPDATE inventory SET quantity=?, status=?, last_update=?, "
                "source=?, amount_mode='level', amount_level=?, "
                "amount_ratio=?, area_px=?, first_in=? WHERE id=?",
                (quantity, status, now, source, level, ratio, area_px,
                 first_in or now, rec_id)
            )
        else:
            old_level = '无'
            self.conn.execute(
                "INSERT INTO inventory "
                "(class_id,class_name,quantity,first_in,last_update,status,"
                "item_type,source,amount_mode,amount_level,amount_ratio,area_px) "
                "VALUES (?,?,?,?,?,?,'fresh',?,'level',?,?,?)",
                (cls_id, name, quantity, now, now, status, source,
                 level, ratio, area_px)
            )

        action_events = {
            'PUT_IN': ('AREA_PUT_IN', '放入'),
            'TAKE_OUT': ('AREA_TAKE_OUT', '取出'),
        }
        if action in action_events:
            event_type, action_text = action_events[action]
            delta_text = (
                f'，面积变化{int(delta_area_px):+d}px'
                if delta_area_px is not None else '')
            note = (
                f'{action_text}{name}，余量：{old_level or "无"}→{level}'
                f'{delta_text}')
            self.conn.execute(
                "INSERT INTO events "
                "(event_time,event_type,class_id,class_name,quantity_change,note) "
                "VALUES (?,?,?,?,?,?)",
                (now, event_type, cls_id, name, 0, note)
            )
        elif old_level != level:
            note = f'{name}余量等级：{old_level or "无"}→{level}'
            self.conn.execute(
                "INSERT INTO events "
                "(event_time,event_type,class_id,class_name,quantity_change,note) "
                "VALUES (?,?,?,?,?,?)",
                (now, 'AREA_LEVEL_CHANGE', cls_id, name, 0, note)
            )
        else:
            note = f'{name}余量等级保持{level}'
        self.conn.commit()
        print(f'[库存] {name} 面积等级: {level} ratio={ratio:.3f} area={area_px}')
        return True, note

    def adjust_area_level(self, class_name, level, ratio=None, area_px=0):
        """人工修正等级型库存，供 App 纠错。"""
        if class_name not in LEVEL_MANAGED_NAMES:
            return False, f'{class_name} 不是等级型库存'
        if level not in LEVEL_DEFAULT_RATIOS:
            return False, f'无效等级: {level}'
        cls_id = CLASSES.index(class_name)
        ratio = LEVEL_DEFAULT_RATIOS[level] if ratio is None else ratio
        return self.set_area_level(
            cls_id, level, ratio, area_px, source='manual_level')

    def _put_in(self, cls_id, qty, now):
        name = CLASSES[cls_id]
        # 检查是否已有该食材在库
        row = self.conn.execute(
            "SELECT id, quantity FROM inventory WHERE class_id=? AND status='in' AND item_type='fresh'",
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
                "INSERT INTO inventory (class_id,class_name,quantity,first_in,last_update,status,item_type,source) VALUES (?,?,?,?,?,'in','fresh','yolo')",
                (cls_id, name, qty, now, now)
            )

        self.conn.execute(
            "INSERT INTO events (event_time,event_type,class_id,class_name,quantity_change,note) VALUES (?,?,?,?,?,?)",
            (now, 'PUT_IN', cls_id, name, qty, f'放入{qty}个{name}')
        )
        self.conn.commit()
        print(f'[库存] 放入: {name} x{qty}')

    def put_package(self, name, qty=1, source='ocr', bbox=None,
                    expire_date=None, category='包装食品', shelf_id='single'):
        """包装物品入库：名称来自 OCR 或用户确认，不依赖 YOLO class_id。"""
        name = (name or '').strip()
        if not name:
            return False, '包装物品名称不能为空'
        qty = max(1, int(qty))
        now = self._now()
        bx, by, bw, bh = self._normalize_bbox(bbox)
        expire_date = self._normalize_date(expire_date)
        category = (category or '包装食品').strip()
        shelf_id = (shelf_id or 'single').strip()
        row = self.conn.execute(
            "SELECT id, quantity FROM inventory WHERE class_name=? AND status='in' AND item_type='package'",
            (name,)
        ).fetchone()
        if row:
            if bbox is None:
                self.conn.execute(
                    "UPDATE inventory SET quantity=?, last_update=?, source=?, expire_date=?, category=?, shelf_id=? WHERE id=?",
                    (row[1] + qty, now, source, expire_date, category,
                     shelf_id, row[0])
                )
            else:
                self.conn.execute(
                    "UPDATE inventory SET quantity=?, last_update=?, source=?, bbox_x=?, bbox_y=?, bbox_w=?, bbox_h=?, expire_date=?, category=?, shelf_id=? WHERE id=?",
                    (row[1] + qty, now, source, bx, by, bw, bh, expire_date,
                     category, shelf_id, row[0])
                )
        else:
            self.conn.execute(
                "INSERT INTO inventory (class_id,class_name,quantity,first_in,last_update,status,item_type,source,bbox_x,bbox_y,bbox_w,bbox_h,expire_date,category,shelf_id) VALUES (-1,?,?,?,?, 'in','package',?,?,?,?,?,?,?,?)",
                (name, qty, now, now, source, bx, by, bw, bh, expire_date,
                 category, shelf_id)
            )
        note = f'包装物品入库：{name} x{qty}'
        self.conn.execute(
            "INSERT INTO events (event_time,event_type,class_id,class_name,quantity_change,note) VALUES (?,?,?,?,?,?)",
            (now, 'PACKAGE_PUT_IN', -1, name, qty, note)
        )
        self.conn.commit()
        print(f'[库存] 包装物品入库: {name} x{qty}')
        return True, note

    def _normalize_bbox(self, bbox):
        if bbox is None:
            return None, None, None, None
        x, y, w, h = bbox
        return int(x), int(y), int(w), int(h)

    def _normalize_date(self, value):
        if value in (None, ''):
            return None
        value = str(value).strip()
        try:
            datetime.strptime(value, '%Y-%m-%d')
            return value
        except ValueError:
            return None

    def get_package_locations(self):
        """返回带位置记忆的包装物品，用于主程序匹配取出。"""
        rows = self.conn.execute(
            "SELECT class_name, quantity, bbox_x, bbox_y, bbox_w, bbox_h FROM inventory WHERE status='in' AND item_type='package' AND quantity>0 AND bbox_x IS NOT NULL"
        ).fetchall()
        return [
            {'name': n, 'qty': q, 'bbox': (int(x), int(y), int(w), int(h))}
            for n, q, x, y, w, h in rows
        ]

    def match_package_by_bbox(self, bbox, thresh=PACKAGE_IOU_THRESH):
        """按 IoU 匹配最可能被取出的包装物品。"""
        best, best_iou = None, thresh
        for rec in self.get_package_locations():
            iou = _iou(rec['bbox'], bbox)
            if iou >= best_iou:
                best, best_iou = rec, iou
        return best

    def adjust_package(self, old_name, new_name=None, new_qty=None,
                       expire_date=None, category=None, shelf_id=None):
        """修改包装物品名称和数量，供 App 纠错 OCR 结果。"""
        old_name = (old_name or '').strip()
        new_name = (new_name if new_name is not None else old_name).strip()
        if not old_name:
            return False, '缺少原包装物品名称'
        if not new_name:
            return False, '包装物品名称不能为空'
        if new_qty is None:
            return False, '缺少 qty 字段'

        now = self._now()
        row = self.conn.execute(
            "SELECT id, quantity, expire_date, category, shelf_id FROM inventory WHERE class_name=? AND status='in' AND item_type='package'",
            (old_name,)
        ).fetchone()
        if row is None:
            return False, f'{old_name} 不在包装物品库存中'

        rec_id, old_qty = row[0], row[1]
        new_qty = max(0, int(new_qty))
        status = 'out' if new_qty == 0 else 'in'
        expire_date = (self._normalize_date(expire_date)
                       if expire_date is not None else row[2])
        category = (category if category is not None else row[3]) or '包装食品'
        shelf_id = (shelf_id if shelf_id is not None else row[4]) or 'single'

        if new_name != old_name and new_qty > 0:
            existing = self.conn.execute(
                "SELECT id, quantity FROM inventory WHERE class_name=? AND status='in' AND item_type='package'",
                (new_name,)
            ).fetchone()
            if existing:
                self.conn.execute(
                    "UPDATE inventory SET quantity=?, last_update=?, source='manual', expire_date=?, category=?, shelf_id=? WHERE id=?",
                    (existing[1] + new_qty, now, expire_date, category,
                     shelf_id, existing[0])
                )
                self.conn.execute(
                    "UPDATE inventory SET quantity=0, status='out', last_update=? WHERE id=?",
                    (now, rec_id)
                )
            else:
                self.conn.execute(
                    "UPDATE inventory SET class_name=?, quantity=?, status=?, last_update=?, source='manual', expire_date=?, category=?, shelf_id=? WHERE id=?",
                    (new_name, new_qty, status, now, expire_date, category,
                     shelf_id, rec_id)
                )
        else:
            self.conn.execute(
                "UPDATE inventory SET class_name=?, quantity=?, status=?, last_update=?, source='manual', expire_date=?, category=?, shelf_id=? WHERE id=?",
                (new_name, new_qty, status, now, expire_date, category,
                 shelf_id, rec_id)
            )

        diff = new_qty - old_qty
        note = f'用户修正包装物品：{old_name} {old_qty}→{new_name} {new_qty}'
        self.conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
            (None, now, 'PACKAGE_MANUAL_ADJUST', -1, new_name, diff, note)
        )
        self.conn.commit()
        print(f'[库存] 包装物品修正: {old_name} {old_qty}→{new_name} {new_qty}')
        return True, note

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

            return

        note = f'{"部分取出" if event_type == "PARTIAL_TAKE_OUT" else "取出"}{qty}个{name}'
        self.conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
            (None, now, event_type, cls_id, name, -qty, note)
        )
        self.conn.commit()
        print(f'[库存] 取出: {name} x{qty}')

    def _take_out_package(self, name, qty, now):
        row = self.conn.execute(
            "SELECT id, quantity FROM inventory WHERE class_name=? AND status='in' AND item_type='package'",
            (name,)
        ).fetchone()
        if row:
            new_qty = max(0, row[1] - qty)
            status = 'out' if new_qty == 0 else 'in'
            self.conn.execute(
                "UPDATE inventory SET quantity=?, status=?, last_update=? WHERE id=?",
                (new_qty, status, now, row[0])
            )
            print(f'[库存] 包装物品取出: {name} x{qty}，剩余: {new_qty}')
        else:
            print(f'[库存] 警告: 取出包装物品{name}但库存无记录，跳过')

            return

        note = f'取出包装物品{name} x{qty}'
        self.conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
            (None, now, 'PACKAGE_TAKE_OUT', -1, name, -qty, note)
        )
        self.conn.commit()

    def has_stock(self):
        """DB 里是否已有在库记录（用于判断是否首次启动）"""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM inventory WHERE status='in' AND quantity>0"
        ).fetchone()
        return row[0] > 0

    def adjust_quantity(self, class_name, new_qty):
        """用户手动修正某种食材数量，写 DB + 记事件"""
        if class_name in LEVEL_MANAGED_NAMES:
            return False, f'{class_name} 使用面积等级，请勿按个数修正'
        now = self._now()
        row = self.conn.execute(
            "SELECT id, quantity, class_id FROM inventory WHERE class_name=? AND status='in'",
            (class_name,)
        ).fetchone()
        if row is None:
            return False, f'{class_name} 不在库存中'
        old_qty, cls_id, rec_id = row[1], row[2], row[0]
        new_qty = max(0, int(new_qty))
        if new_qty == old_qty:
            return True, f'{class_name} 数量未变化'
        status = 'out' if new_qty == 0 else 'in'
        self.conn.execute(
            "UPDATE inventory SET quantity=?, status=?, last_update=? WHERE id=?",
            (new_qty, status, now, rec_id)
        )
        diff = new_qty - old_qty
        note = f'用户手动修正：{class_name} {old_qty}→{new_qty}'
        self.conn.execute(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
            (None, now, 'MANUAL_ADJUST', cls_id, class_name, diff, note)
        )
        self.conn.commit()
        print(f'[库存] 手动修正: {class_name} {old_qty}→{new_qty}')
        return True, note

    def get_current_stock(self):
        """获取当前在库食材"""
        rows = self.conn.execute(
            "SELECT class_name, quantity, first_in, last_update FROM inventory "
            "WHERE status='in' AND (quantity>0 OR "
            "(amount_mode='level' AND amount_level!='无')) "
            "ORDER BY last_update DESC"
        ).fetchall()
        return rows

    def get_current_stock_records(self):
        """获取当前库存，包含 App 可用于分组展示的类型字段。"""
        rows = self.conn.execute(
            "SELECT class_name, quantity, first_in, last_update, item_type, "
            "source, expire_date, category, shelf_id, amount_mode, "
            "amount_level, amount_ratio, area_px FROM inventory "
            "WHERE status='in' AND (quantity>0 OR "
            "(amount_mode='level' AND amount_level!='无')) "
            "ORDER BY last_update DESC"
        ).fetchall()
        return [
            {'name': n, 'qty': q, 'first_in': f, 'last_update': l,
             'type': item_type or 'fresh', 'source': source or 'yolo',
             'expire_date': expire_date,
             'category': category or ('包装食品' if item_type == 'package'
                                      else '生鲜食材'),
             'shelf_id': shelf_id or 'single',
             'days_to_expire': self._days_to_expire(expire_date),
             'expire_status': self._expire_status(expire_date),
             'amount_mode': amount_mode or 'count',
             'amount_level': amount_level,
             'amount_ratio': amount_ratio,
             'area_px': area_px,
             'display_amount': (amount_level if amount_mode == 'level'
                                else f'{q}个')}
            for n, q, f, l, item_type, source, expire_date, category, shelf_id,
            amount_mode, amount_level, amount_ratio, area_px
            in rows
        ]

    def _days_to_expire(self, expire_date):
        if not expire_date:
            return None
        try:
            d = datetime.strptime(expire_date, '%Y-%m-%d').date()
        except ValueError:
            return None
        return (d - date.today()).days

    def _expire_status(self, expire_date):
        days = self._days_to_expire(expire_date)
        if days is None:
            return 'none'
        if days < 0:
            return 'expired'
        if days <= 3:
            return 'soon'
        return 'normal'

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
            if name in LEVEL_MANAGED_NAMES:
                row = self.conn.execute(
                    "SELECT amount_level FROM inventory WHERE class_name=? "
                    "AND status='in' ORDER BY id DESC LIMIT 1", (name,)
                ).fetchone()
                print(f'  {name}: {row[0] if row else "未知"} | '
                      f'入库:{first_in} | 更新:{last_update}')
            else:
                print(f'  {name}: {qty}个 | 入库:{first_in} | 更新:{last_update}')
        print('======================\n')

    def close(self):
        self.conn.close()
