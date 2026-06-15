import numpy as np
from motion import changed_pixel_count, is_moving, is_stable_window


def test_identical_frames_not_moving():
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    assert is_moving(frame, frame) is False


def test_large_change_is_moving():
    prev = np.full((480, 640, 3), 120, dtype=np.uint8)
    cur = prev.copy()
    cur[100:300, 100:400] = 255  # 一大块区域明显变化
    assert is_moving(prev, cur) is True


def test_tiny_noise_not_moving():
    prev = np.full((480, 640, 3), 120, dtype=np.uint8)
    cur = prev.copy()
    cur[0:5, 0:5] = 255  # 极小噪点，不算运动
    assert is_moving(prev, cur) is False


def test_stable_window_rejects_slow_cumulative_change():
    frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(3)]
    frames[-1][:] = 255
    assert is_stable_window(frames, lambda a, b: changed_pixel_count(a, b) > 500)
    assert is_stable_window(
        frames, lambda a, b: changed_pixel_count(a, b) > 10) is False
