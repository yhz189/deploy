import numpy as np

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
