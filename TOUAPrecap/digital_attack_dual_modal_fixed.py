import matplotlib.pyplot as plt
import os
from detect_single_image import yolov3_vis, yolov3_inf
import random
import cv2
import numpy as np
from functions import initiation_8, octagon_inf, clip, initiation_color_18, octagon_vis1

ASR = [0, 0, 0, 0, 0, 0]
Query = [0, 0, 0, 0, 0, 0]
count_all = [0, 0, 0, 0, 0, 0]

detector_inf = yolov3_inf
detector_vis = yolov3_vis

for net_id in range(0, 1):
    dir_inf = r'dataset_attack/inf_' + str(net_id) + '/'
    dir_vis = r'dataset_attack/vis_' + str(net_id) + '/'

    file = os.listdir(dir_inf)
    tag = 0
    for pic in file:
        tag = tag + 1

        img_inf_path = dir_inf + pic
        img_vis_path = dir_vis + pic

        res_inf1 = detector_inf(img_inf_path)
        res_vis1 = detector_vis(img_vis_path)

        # Skip if no detection
        if res_inf1[0].shape[0] == 0 or res_vis1[0].shape[0] == 0:
            print(f'Skipping {pic}: No detection (inf={res_inf1[0].shape[0]}, vis={res_vis1[0].shape[0]})')
            continue

        shape_inf, b1 = res_inf1[0].shape
        shape_vis, b2 = res_vis1[0].shape

        print('net_id, pic_id, shape_inf, shape_vis = ', net_id, tag, shape_inf, shape_vis)
        print('ASR = ', ASR)
        print('Query = ', Query)
        print('count_all = ', count_all)

        if shape_vis != 1 or shape_inf != 1:
            continue

        if res_vis1[0][0][0] > (res_inf1[0][0][0] + ((res_inf1[0][0][2] - res_inf1[0][0][0]) / 2)) or res_inf1[0][0][0] > (
                res_vis1[0][0][0] + ((res_vis1[0][0][2] - res_vis1[0][0][0]) / 2)) or res_vis1[0][0][1] > (
                res_inf1[0][0][1] + ((res_inf1[0][0][3] - res_inf1[0][0][1]) / 2)) or res_inf1[0][0][1] > (
                res_vis1[0][0][1] + ((res_vis1[0][0][3] - res_vis1[0][0][1]) / 2)):
            continue

        count_all[net_id] = count_all[net_id] + 1
        print(f'Processing image {tag}/{len(file)}: {pic}')
        
        # Rest of attack code here
        break

print('ASR = ', ASR)
print('Query = ', Query)
print('count_all = ', count_all)
