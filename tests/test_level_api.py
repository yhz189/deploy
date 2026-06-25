import inventory
import web_server


def test_stock_api_returns_level_fields_and_supports_manual_level(tmp_path):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    client = web_server.app.test_client()

    response = client.post('/api/stock/adjust-level', json={
        'name': 'banana',
        'level': '适量',
        'ratio': 0.51,
        'area_px': 12000,
    })
    assert response.status_code == 200

    stock = client.get('/api/stock').get_json()
    assert stock[0]['name'] == 'banana'
    assert stock[0]['amount_mode'] == 'level'
    assert stock[0]['amount_level'] == '适量'
    assert stock[0]['display_amount'] == '适量'
    assert stock[0]['amount_ratio'] == 0.51

    response = client.post('/api/stock/adjust', json={
        'name': 'banana',
        'qty': 5,
    })
    assert response.status_code == 404


def test_level_none_is_hidden_in_stock_api(tmp_path):
    inventory.DB_PATH = str(tmp_path / 'inventory.db')
    client = web_server.app.test_client()
    response = client.post('/api/stock/adjust-level', json={
        'name': 'carrot',
        'level': '无',
    })
    assert response.status_code == 200
    stock = client.get('/api/stock').get_json()
    assert stock == []
