import os
import pickle

from tqdm import tqdm
import cv2
import numpy as np
from paddleocr import PaddleOCR

if __name__ == '__main__':
    ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    pic_folder = './data'
    all_pic_name = os.listdir(pic_folder)
    iterator = tqdm(range(len(all_pic_name)))
    all_data_dict = {}
    for pic_index in iterator:
        pic_name = all_pic_name[pic_index]
        pic_path = os.path.join(pic_folder, pic_name)
        pic = cv2.imread(pic_path)
        ocr_result = ocr.ocr(np.array(pic), cls=False)[0]

        # 处理字符串
        is_ok_data = True
        now_data_dict = {}
        for result_index in range(len(ocr_result)):
            result = ocr_result[result_index]
            result_str = result[1][0]

            if '=' in result_str:
                para_name, para_value = result_str.split('=')
                # 去除na
                if para_value == "NA":
                    is_ok_data = False
                    break
                else:
                    now_data_dict[para_name] = para_value
            elif '~' in result_str:
                is_ok_data = False
                break

        if is_ok_data:
            all_data_dict[pic_index] = now_data_dict

    result_path = './pic_data_dict.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(all_data_dict, f)
