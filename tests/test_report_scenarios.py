import numpy as np

from change_locator import classify_regions, find_change_regions


def _ident(class_id, fine, coarse):
    return {
        "class_id": class_id,
        "fine": fine,
        "coarse": coarse,
        "conf": 0.9,
        "area": 1000.0,
    }


def test_report_scenario_replace_is_decided_after_before_after_classification():
    ref = np.zeros((80, 120, 3), dtype=np.uint8)
    new = np.zeros_like(ref)
    ref[:, :, 0] = 10
    new[:, :, 0] = 20
    apple = _ident(0, "Apple", "fruit")
    banana = _ident(1, "Banana", "produce")

    def classify(crop):
        return apple if int(crop[0, 0, 0]) == 10 else banana

    regions = classify_regions(ref, new, [(5, 7, 70, 40)], classify)

    assert regions[0].bbox == (5, 7, 70, 40)
    assert regions[0].kind == "REPLACE"
    assert regions[0].ref_ident == apple
    assert regions[0].new_ident == banana


def test_report_scenario_same_is_decided_after_before_after_classification():
    ref = np.full((80, 120, 3), 10, dtype=np.uint8)
    new = np.full_like(ref, 20)
    apple_before = _ident(0, "Apple", "fruit")
    apple_after = _ident(0, "Apple", "fruit")

    def classify(crop):
        return apple_before if int(crop[0, 0, 0]) == 10 else apple_after

    regions = classify_regions(ref, new, [(8, 9, 60, 35)], classify)

    assert regions[0].kind == "SAME"
    assert regions[0].ref_ident == apple_before
    assert regions[0].new_ident == apple_after


def test_report_scenario_synthetic_frames_flow_from_opencv_region_to_appear():
    ref = np.full((480, 640, 3), 100, dtype=np.uint8)
    new = ref.copy()
    new[180:300, 240:380] = 230
    banana = _ident(1, "Banana", "fruit")

    bboxes = find_change_regions(ref, new)
    assert len(bboxes) == 1
    x, y, w, h = bboxes[0]
    assert w != h
    assert 0 <= x < x + w <= 640
    assert 0 <= y < y + h <= 480

    def classify(crop):
        return banana if int(crop.max()) > 200 else None

    regions = classify_regions(ref, new, bboxes, classify)

    assert regions[0].kind == "APPEAR"
    assert regions[0].new_ident == banana
