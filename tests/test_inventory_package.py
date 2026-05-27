import inventory
from inventory import InventoryManager


def test_put_package_creates_package_stock_record(tmp_path):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    inv = InventoryManager()
    try:
        ok, msg = inv.put_package('纯牛奶', qty=2)
        assert ok is True
        assert '纯牛奶' in msg

        records = inv.get_current_stock_records()
        assert records[0]['name'] == '纯牛奶'
        assert records[0]['qty'] == 2
        assert records[0]['type'] == 'package'
        assert records[0]['source'] == 'ocr'
        assert records[0]['category'] == '包装食品'
        assert records[0]['shelf_id'] == 'single'
        assert records[0]['expire_status'] == 'none'

        events = inv.get_recent_events(1)
        assert events[0][1] == 'PACKAGE_PUT_IN'
        assert '包装物品入库' in events[0][2]
    finally:
        inv.close()


def test_put_package_merges_same_name(tmp_path):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    inv = InventoryManager()
    try:
        inv.put_package('每日坚果', qty=1)
        inv.put_package('每日坚果', qty=2, source='ocr_confirmed')
        records = inv.get_current_stock_records()
        assert len(records) == 1
        assert records[0]['name'] == '每日坚果'
        assert records[0]['qty'] == 3
        assert records[0]['type'] == 'package'
        assert records[0]['source'] == 'ocr_confirmed'
    finally:
        inv.close()


def test_adjust_package_renames_and_updates_qty(tmp_path):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    inv = InventoryManager()
    try:
        inv.put_package('纯牛奶', qty=1)
        ok, msg = inv.adjust_package('纯牛奶', '蒙牛纯牛奶', 3)
        assert ok is True
        assert '蒙牛纯牛奶' in msg

        records = inv.get_current_stock_records()
        assert len(records) == 1
        assert records[0]['name'] == '蒙牛纯牛奶'
        assert records[0]['qty'] == 3
        assert records[0]['type'] == 'package'
        assert records[0]['source'] == 'manual'

        events = inv.get_recent_events(1)
        assert events[0][1] == 'PACKAGE_MANUAL_ADJUST'
    finally:
        inv.close()


def test_put_package_stores_bbox_and_matches_takeout(tmp_path):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    inv = InventoryManager()
    try:
        inv.put_package('火锅底料', qty=1, bbox=(20, 30, 80, 90))
        match = inv.match_package_by_bbox((24, 34, 78, 88))
        assert match is not None
        assert match['name'] == '火锅底料'

        inv.process_event('PACKAGE_TAKE_OUT',
                          {'removed_package': {'火锅底料': 1}})
        records = inv.get_current_stock_records()
        assert records == []

        events = inv.get_recent_events(1)
        assert events[0][1] == 'PACKAGE_TAKE_OUT'
        assert '火锅底料' in events[0][2]
    finally:
        inv.close()


def test_put_package_records_expiry_and_cloud_source(tmp_path):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    inv = InventoryManager()
    try:
        inv.put_package('酸奶', qty=1, source='cloud',
                        expire_date='2099-01-02',
                        category='饮料乳品', shelf_id='middle')
        records = inv.get_current_stock_records()
        assert records[0]['name'] == '酸奶'
        assert records[0]['source'] == 'cloud'
        assert records[0]['expire_date'] == '2099-01-02'
        assert records[0]['category'] == '饮料乳品'
        assert records[0]['shelf_id'] == 'middle'
        assert records[0]['expire_status'] == 'normal'

        ok, msg = inv.adjust_package('酸奶', '低温酸奶', 2,
                                     expire_date='2026-06-20',
                                     category='乳制品', shelf_id='upper')
        assert ok is True
        records = inv.get_current_stock_records()
        assert records[0]['name'] == '低温酸奶'
        assert records[0]['qty'] == 2
        assert records[0]['category'] == '乳制品'
        assert records[0]['shelf_id'] == 'upper'
    finally:
        inv.close()
