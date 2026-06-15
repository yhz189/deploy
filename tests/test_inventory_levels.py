import inventory
import sqlite3
from inventory import InventoryManager
from utils import CLASSES


def test_banana_and_carrot_use_absolute_area_levels(tmp_path):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    inv = InventoryManager()
    banana_id = CLASSES.index('banana')
    carrot_id = CLASSES.index('carrot')
    try:
        inv.process_event('PUT_IN', {'added': {banana_id: 3, carrot_id: 2}})
        assert inv.get_current_stock_records() == []

        inv.set_area_level(banana_id, '适量', 0.52, 12345)
        inv.set_area_level(carrot_id, '少量', 0.18, 4321)
        records = {row['name']: row for row in inv.get_current_stock_records()}

        assert records['banana']['amount_mode'] == 'level'
        assert records['banana']['amount_level'] == '适量'
        assert records['banana']['display_amount'] == '适量'
        assert records['banana']['amount_ratio'] == 0.52
        assert records['banana']['area_px'] == 12345
        assert records['carrot']['display_amount'] == '少量'

        inv.process_event('TAKE_OUT', {'removed': {banana_id: 1}})
        assert {row['name']: row for row in inv.get_current_stock_records()}[
            'banana']['amount_level'] == '适量'
    finally:
        inv.close()


def test_area_level_none_removes_item_and_records_level_change(tmp_path):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    inv = InventoryManager()
    banana_id = CLASSES.index('banana')
    try:
        inv.set_area_level(banana_id, '大量', 0.8, 20000)
        inv.set_area_level(banana_id, '无', 0.0, 0)
        records = inv.get_current_stock_records()
        assert records[0]['amount_level'] == '无'
        assert records[0]['display_amount'] == '无'
        assert records[0]['qty'] == 0
        events = inv.get_recent_events(2)
        assert events[0][1] == 'AREA_LEVEL_CHANGE'
        assert '大量→无' in events[0][2]
    finally:
        inv.close()


def test_banana_area_action_records_direction_even_with_same_level(tmp_path):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    inv = InventoryManager()
    banana_id = CLASSES.index('banana')
    try:
        inv.set_area_level(banana_id, '少量', 0.18, 5000)
        inv.set_area_level(
            banana_id, '少量', 0.25, 8000,
            action='PUT_IN', delta_area_px=3000)
        inv.set_area_level(
            banana_id, '少量', 0.20, 6000,
            action='TAKE_OUT', delta_area_px=-2000)
        events = inv.get_recent_events(2)
        assert events[0][1] == 'AREA_TAKE_OUT'
        assert '取出banana' in events[0][2]
        assert events[1][1] == 'AREA_PUT_IN'
        assert '放入banana' in events[1][2]
    finally:
        inv.close()


def test_confirmed_area_put_in_cannot_remain_none(tmp_path):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    inv = InventoryManager()
    banana_id = CLASSES.index('banana')
    try:
        inv.set_area_level(
            banana_id, '无', 0.08, 9000,
            action='PUT_IN', delta_area_px=9000)
        record = inv.get_current_stock_records()[0]
        assert record['amount_level'] == '少量'
        assert record['amount_ratio'] == 0.18
        assert '无→少量' in inv.get_recent_events(1)[0][2]
    finally:
        inv.close()


def test_count_adjustment_rejected_for_level_managed_food(tmp_path):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    inv = InventoryManager()
    try:
        ok, msg = inv.adjust_quantity('banana', 3)
        assert ok is False
        assert '面积等级' in msg
    finally:
        inv.close()


def test_existing_count_rows_are_migrated_to_level_mode(tmp_path):
    path = tmp_path / 'inventory.db'
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE inventory (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "class_id INTEGER NOT NULL, class_name TEXT NOT NULL, "
        "quantity INTEGER DEFAULT 1, first_in TEXT, last_update TEXT, "
        "status TEXT DEFAULT 'in', item_type TEXT DEFAULT 'fresh', "
        "source TEXT DEFAULT 'yolo')")
    conn.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "event_time TEXT, event_type TEXT, class_id INTEGER, "
        "class_name TEXT, quantity_change INTEGER, note TEXT)")
    conn.execute(
        "INSERT INTO inventory "
        "(class_id,class_name,quantity,status,item_type) VALUES (2,'banana',3,'in','fresh')")
    conn.commit()
    conn.close()

    inventory.DB_PATH = str(path)
    inv = InventoryManager()
    try:
        record = inv.get_current_stock_records()[0]
        assert record['amount_mode'] == 'level'
        assert record['amount_level'] == '大量'
        assert record['qty'] == 1
    finally:
        inv.close()


def test_duplicate_level_rows_are_consolidated(tmp_path):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    inv = InventoryManager()
    banana_id = CLASSES.index('banana')
    try:
        inv.set_area_level(banana_id, '少量', 0.2, 100)
        inv.conn.execute(
            "INSERT INTO inventory "
            "(class_id,class_name,quantity,status,item_type,amount_mode,"
            "amount_level,amount_ratio,area_px) "
            "VALUES (?, 'banana', 1, 'in', 'fresh', 'level', '大量', 0.8, 500)",
            (banana_id,))
        inv.conn.commit()
    finally:
        inv.close()

    inv = InventoryManager()
    try:
        rows = inv.conn.execute(
            "SELECT status FROM inventory WHERE class_name='banana'"
        ).fetchall()
        assert sum(status == 'in' for (status,) in rows) == 1
        assert len(inv.get_current_stock_records()) == 1
    finally:
        inv.close()


def test_noop_count_adjustment_does_not_record_event(tmp_path):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    inv = InventoryManager()
    apple_id = CLASSES.index('apple')
    try:
        inv.process_event('PUT_IN', {'added': {apple_id: 1}})
        before = len(inv.get_recent_events(10))
        ok, _ = inv.adjust_quantity('apple', 1)
        assert ok is True
        assert len(inv.get_recent_events(10)) == before
    finally:
        inv.close()
