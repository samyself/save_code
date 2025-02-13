import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
import pyautogui
import pyperclip
import time
from pynput.keyboard import Key, Controller
import http.client
import os
from PIL import ImageGrab

from tqdm import tqdm



if __name__ == '__main__':
    pic_folder = './data'
    # 设置pyautogui的失败安全开关，执行动作太快时可以及时中断
    pyautogui.FAILSAFE = False

    # 获取屏幕尺寸
    screen_width, screen_height = pyautogui.size()
    controller = Controller()
    print(screen_width, screen_height)

    # 定义截屏范围的左上角和右下角坐标
    pic_left_up_x_related_place = 0.75
    pic_left_up_y_related_place = 0.94
    pic_right_down_x_related_place = 1
    pic_right_down_y_related_place = 0.96
    bbox = (screen_width * pic_left_up_x_related_place, screen_height * pic_left_up_y_related_place,
            screen_width * pic_right_down_x_related_place, screen_height * pic_right_down_y_related_place)

    # 鼠标范围
    point_related_min_x = 0.05
    point_related_max_x = 0.7
    point_related_step_x = 0.01
    point_related_all_x = np.arange(point_related_min_x, point_related_max_x, point_related_step_x)

    point_related_min_y = 0.15
    point_related_max_y = 0.85
    point_related_step_y = 0.01
    point_related_all_y = np.arange(point_related_min_y, point_related_max_y, point_related_step_y)

    pic_index = 0
    iterator_x = tqdm(range(len(point_related_all_x)))
    iterator_y = tqdm(range(len(point_related_all_y)))

    # ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    for x_index in iterator_x:
        point_related_x = point_related_all_x[x_index]
        for y_index in iterator_y:
            point_related_y = point_related_all_y[y_index]

            # 鼠标移动导对应位置
            pyautogui.moveTo(screen_width * point_related_x, screen_height * point_related_y, duration=0.1)

            # 捕捉指定范围的屏幕
            screenshot = ImageGrab.grab(bbox=bbox)
            # ocr_result = ocr.ocr(np.array(screenshot), cls=False)
            # 保存截图到文件
            screenshot_path = f"{pic_folder}/{x_index}_{y_index}_screenshot.png"
            screenshot.save(screenshot_path)
            pic_index += 1
