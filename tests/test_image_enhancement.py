import cv2
import numpy as np

from image_enhancement import (analyze_complex_conditions,
                               detect_highlights, enhance_low_light,
                               reflective_cover_risk, suppress_highlights)


def test_normal_light_is_not_low_light_enhanced():
    image = np.full((80, 120, 3), 150, dtype=np.uint8)
    enhanced, result = enhance_low_light(image)
    assert result['applied'] is False
    assert np.array_equal(enhanced, image)


def test_low_light_enhancement_increases_mean_brightness():
    image = np.full((80, 120, 3), 25, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 60), (45, 45, 45), -1)
    _, result = enhance_low_light(image)
    assert result['applied'] is True
    assert result['after']['mean_brightness'] > result['before']['mean_brightness']


def test_highlight_detection_and_suppression_reduce_brightness():
    image = np.full((100, 140, 3), (30, 80, 120), dtype=np.uint8)
    cv2.circle(image, (70, 50), 15, (255, 255, 255), -1)
    mask, stats = detect_highlights(image)
    suppressed = suppress_highlights(image, mask)
    assert stats['area_px'] > 400
    assert stats['area_ratio'] > 0
    assert mask.shape == image.shape[:2]
    assert suppressed.shape == image.shape
    assert suppressed[50, 70].mean() < image[50, 70].mean()


def test_reflective_cover_risk_has_safe_output_contract():
    image = np.zeros((100, 140, 3), dtype=np.uint8)
    for x in range(10, 130, 20):
        cv2.rectangle(image, (x, 20), (x + 8, 80), (255, 255, 255), -1)
    mask, stats = detect_highlights(image)
    result = reflective_cover_risk(image, mask, stats)
    assert set(result) == {
        'possible_reflective_cover', 'risk_score',
        'edge_density_near_highlights'}
    assert 0.0 <= result['risk_score'] <= 1.0


def test_full_analysis_returns_all_requested_images_and_limitations():
    image = np.full((60, 80, 3), 40, dtype=np.uint8)
    images, result = analyze_complex_conditions(image)
    assert set(images) == {
        'low_light_enhanced', 'highlight_mask',
        'highlight_suppressed'}
    assert result['mode'] == 'offline_demonstration'
    assert len(result['limitations']) == 3
