import numpy as np
from motion import is_moving


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
