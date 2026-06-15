import numpy as np
import pytest

import package_ocr


def test_save_and_load_package_takeout_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(package_ocr, 'PACKAGE_TAKEOUT_REF_PATH',
                        str(tmp_path / 'ref.jpg'))
    monkeypatch.setattr(package_ocr, 'PACKAGE_TAKEOUT_NEW_PATH',
                        str(tmp_path / 'new.jpg'))
    monkeypatch.setattr(package_ocr, 'PACKAGE_TAKEOUT_META_PATH',
                        str(tmp_path / 'takeout.json'))

    ref = np.full((20, 30, 3), 120, dtype=np.uint8)
    new = np.full((20, 30, 3), 200, dtype=np.uint8)
    cand = package_ocr.save_package_takeout_candidate(
        ref, new, bbox=(1, 2, 30, 20), reason='unit-test')

    assert cand['engine'] == 'phone_ocr'
    assert cand['bbox'] == (1, 2, 30, 20)
    loaded = package_ocr.load_package_takeout_candidate()
    assert loaded['reason'] == 'unit-test'
    assert loaded['ref_image_url'] == '/package/takeout/ref_image'
    assert loaded['new_image_url'] == '/package/takeout/new_image'

    package_ocr.clear_package_takeout_candidate()
    assert package_ocr.load_package_takeout_candidate() is None


def _set_candidate_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(package_ocr, 'PACKAGE_CROP_PATH',
                        str(tmp_path / 'candidate.jpg'))
    monkeypatch.setattr(package_ocr, 'PACKAGE_META_PATH',
                        str(tmp_path / 'candidate.json'))


def test_package_candidate_has_id_and_expires(tmp_path, monkeypatch):
    _set_candidate_paths(tmp_path, monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(package_ocr.time, 'time', lambda: now[0])
    crop = np.full((20, 30, 3), 120, dtype=np.uint8)
    cand = package_ocr.save_package_candidate(
        crop, {'ok': False, 'engine': 'phone_ocr'}, bbox=(1, 2, 3, 4))
    assert cand['candidate_id']
    assert cand['status'] == 'pending_ocr'
    assert cand['expires_epoch'] == 1060.0

    now[0] = 1061.0
    assert package_ocr.load_package_candidate() is None
    assert not (tmp_path / 'candidate.jpg').exists()


@pytest.mark.parametrize('text', ['', '1', '250ml', '***', 'A', '\ufffd\ufffd'])
def test_invalid_package_text_is_rejected(text):
    assert package_ocr.is_valid_package_text(text) is False


def test_package_confirmation_requires_matching_id_confidence_and_text(
        tmp_path, monkeypatch):
    _set_candidate_paths(tmp_path, monkeypatch)
    crop = np.full((20, 30, 3), 120, dtype=np.uint8)
    cand = package_ocr.save_package_candidate(
        crop, {'ok': False, 'engine': 'phone_ocr'})

    ok, _, _ = package_ocr.validate_package_confirmation(
        'wrong', '纯牛奶', 0.9)
    assert ok is False
    ok, _, _ = package_ocr.validate_package_confirmation(
        cand['candidate_id'], '纯牛奶', 0.4)
    assert ok is False
    ok, _, _ = package_ocr.validate_package_confirmation(
        cand['candidate_id'], '250ml', 0.9)
    assert ok is False
    ok, _, _ = package_ocr.validate_package_confirmation(
        cand['candidate_id'], '纯牛奶', 0.9)
    assert ok is True


def test_package_candidate_can_only_be_consumed_once(tmp_path, monkeypatch):
    _set_candidate_paths(tmp_path, monkeypatch)
    crop = np.full((20, 30, 3), 120, dtype=np.uint8)
    cand = package_ocr.save_package_candidate(
        crop, {'ok': False, 'engine': 'phone_ocr'})
    ok, _, _ = package_ocr.consume_package_candidate(
        cand['candidate_id'], '纯牛奶', 0.9)
    assert ok is True
    ok, _, _ = package_ocr.consume_package_candidate(
        cand['candidate_id'], '纯牛奶', 0.9)
    assert ok is False
