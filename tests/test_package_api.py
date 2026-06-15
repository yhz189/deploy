import numpy as np

import inventory
import package_ocr
import web_server


def _configure(tmp_path, monkeypatch):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    monkeypatch.setattr(package_ocr, 'PACKAGE_CROP_PATH',
                        str(tmp_path / 'candidate.jpg'))
    monkeypatch.setattr(package_ocr, 'PACKAGE_META_PATH',
                        str(tmp_path / 'candidate.json'))
    return web_server.app.test_client()


def _save_candidate():
    crop = np.full((30, 40, 3), 120, dtype=np.uint8)
    return package_ocr.save_package_candidate(
        crop, {'ok': False, 'engine': 'phone_ocr'}, bbox=(1, 2, 30, 20))


def test_confirm_consumes_candidate_and_cannot_write_twice(tmp_path, monkeypatch):
    client = _configure(tmp_path, monkeypatch)
    cand = _save_candidate()
    payload = {
        'candidate_id': cand['candidate_id'],
        'name': '纯牛奶',
        'confidence': 0.91,
        'source': 'app_ocr',
        'qty': 1,
    }
    assert client.post('/api/package/confirm', json=payload).status_code == 200
    assert client.post('/api/package/confirm', json=payload).status_code == 409

    inv = inventory.InventoryManager()
    try:
        records = inv.get_current_stock_records()
        assert len(records) == 1
        assert records[0]['name'] == '纯牛奶'
        assert records[0]['qty'] == 1
    finally:
        inv.close()


def test_delete_requires_matching_candidate_id(tmp_path, monkeypatch):
    client = _configure(tmp_path, monkeypatch)
    cand = _save_candidate()
    assert client.delete(
        '/api/package/candidate?candidate_id=wrong').status_code == 409
    assert client.delete(
        f'/api/package/candidate?candidate_id={cand["candidate_id"]}'
    ).status_code == 200
    assert client.get('/api/package/candidate').status_code == 404


def test_confirm_rejects_low_quality_ocr(tmp_path, monkeypatch):
    client = _configure(tmp_path, monkeypatch)
    cand = _save_candidate()
    response = client.post('/api/package/confirm', json={
        'candidate_id': cand['candidate_id'],
        'name': '250ml',
        'confidence': 0.95,
    })
    assert response.status_code == 409
    assert client.get('/api/package/candidate').status_code == 200
