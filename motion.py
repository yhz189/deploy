"""运动门控：判断两帧之间是否有运动"""
import cv2


def is_moving(prev_bgr, cur_bgr, diff_thresh=25, area_thresh=3000):
    """两帧灰度差分，运动像素数超过 area_thresh 即判定有运动"""
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, cur_gray)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    return int(cv2.countNonZero(mask)) > area_thresh
