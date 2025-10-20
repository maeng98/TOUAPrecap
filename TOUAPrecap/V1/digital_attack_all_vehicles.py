"""
모든 차량을 각각 공격하는 버전
"""
import matplotlib.pyplot as plt
import os
from detect_single_image_yolov5 import yolov5_vis, yolov5_inf
import random
import cv2
import numpy as np
from functions import initiation_8, octagon_inf, clip, initiation_color_18, octagon_vis1

ASR = [0]
Query = [0]
count_all = [0]
total_vehicles = 0
attacked_vehicles = 0

detector_inf = yolov5_inf
detector_vis = yolov5_vis

net_id = 0

dir_inf = r'datasets/test/infrared/'
dir_vis = r'datasets/test/visible/'

file = os.listdir(dir_inf)
tag = 0

for pic in file:
    tag = tag + 1
    
    img_inf_path = dir_inf + pic
    img_vis_path = dir_vis + pic
    
    # 초기 탐지
    res_inf1 = detector_inf(img_inf_path)
    res_vis1 = detector_vis(img_vis_path)
    
    car_class_idx = 2  # 차량 클래스 인덱스
    
    shape_inf = len(res_inf1[0][car_class_idx]) if len(res_inf1[0][car_class_idx]) > 0 else 0
    shape_vis = len(res_vis1[0][car_class_idx]) if len(res_vis1[0][car_class_idx]) > 0 else 0
    
    print(f'\n{"="*60}')
    print(f'이미지: {pic} (pic_id={tag})')
    print(f'적외선 차량: {shape_inf}대, 가시광선 차량: {shape_vis}대')
    
    if shape_vis == 0 or shape_inf == 0:
        print('차량 없음, 건너뛰기')
        continue
    
    # 모든 차량 쌍을 찾기 (적외선-가시광선 매칭)
    vehicle_pairs = []
    
    for i, bbox_inf in enumerate(res_inf1[0][car_class_idx]):
        # 적외선 차량의 중심점
        center_x_inf = (bbox_inf[0] + bbox_inf[2]) / 2
        center_y_inf = (bbox_inf[1] + bbox_inf[3]) / 2
        
        # 가장 가까운 가시광선 차량 찾기
        min_distance = float('inf')
        matched_vis_idx = -1
        
        for j, bbox_vis in enumerate(res_vis1[0][car_class_idx]):
            center_x_vis = (bbox_vis[0] + bbox_vis[2]) / 2
            center_y_vis = (bbox_vis[1] + bbox_vis[3]) / 2
            
            distance = np.sqrt((center_x_inf - center_x_vis)**2 + 
                             (center_y_inf - center_y_vis)**2)
            
            if distance < min_distance and distance < 150:  # 150픽셀 이내
                min_distance = distance
                matched_vis_idx = j
        
        if matched_vis_idx != -1:
            vehicle_pairs.append({
                'inf_idx': i,
                'vis_idx': matched_vis_idx,
                'bbox_inf': bbox_inf,
                'bbox_vis': res_vis1[0][car_class_idx][matched_vis_idx],
                'distance': min_distance
            })
    
    print(f'매칭된 차량 쌍: {len(vehicle_pairs)}개')
    
    if len(vehicle_pairs) == 0:
        print('매칭된 차량 없음, 건너뛰기')
        continue
    
    # 각 차량 쌍에 대해 공격 시도
    for pair_idx, pair in enumerate(vehicle_pairs):
        total_vehicles += 1
        
        print(f'\n--- 차량 {pair_idx + 1}/{len(vehicle_pairs)} 공격 시작 ---')
        print(f'거리: {pair["distance"]:.1f}px')
        
        bbox_inf = pair['bbox_inf']
        bbox_vis = pair['bbox_vis']
        
        count_all[net_id] = count_all[net_id] + 1
        
        # PSO 파라미터
        D = 18
        omega, c1, r1, c2, r2 = 0.9, 1.6, 0.5, 1.4, 0.5
        seed = 50  # 여러 차량 공격이므로 seed 줄임
        step = 5   # step도 줄임
        shape = 8
        Gein_inf = 2 * shape
        
        population = np.zeros((seed, Gein_inf))
        
        X1, Y1, X2, Y2 = int(bbox_inf[0]), int(bbox_inf[1]), int(bbox_inf[2]), int(bbox_inf[3])
        
        # 바운딩 박스가 너무 작으면 건너뛰기
        if (X2 - X1) < 30 or (Y2 - Y1) < 30:
            print(f'바운딩 박스 너무 작음 ({X2-X1}x{Y2-Y1}), 건너뛰기')
            continue
        
        population = initiation_8(population, X1, Y1, X2, Y2)
        conf = np.zeros((1, seed))
        P_best = np.zeros((seed, Gein_inf))
        conf_p = np.ones((1, seed)) * 100
        G_best = np.zeros((1, Gein_inf))
        conf_G = 100
        V = np.zeros((seed, Gein_inf))
        
        print(f'바운딩 박스: X1={X1}, Y1={Y1}, X2={X2}, Y2={Y2}')
        
        path_adv_vis = f'adv_vis_{pair_idx}.jpg'
        path_adv_inf = f'adv_inf_{pair_idx}.jpg'
        
        tag_break = 0
        tag_run_vis = 0
        
        # 적외선 공격
        for steps in range(step):
            if tag_break == 1:
                break
                
            for seeds in range(seed):
                Query[net_id] = Query[net_id] + 1
                
                unit = population[seeds]
                img_inf = cv2.imread(img_inf_path)
                
                octagon_inf(img_inf, unit, path_adv_inf)
                res_inf = detector_inf(path_adv_inf)
                
                # 이 특정 차량이 탐지되지 않았는지 확인
                detected = False
                if len(res_inf[0][car_class_idx]) > 0:
                    for det_bbox in res_inf[0][car_class_idx]:
                        # IoU 계산으로 같은 차량인지 확인
                        x1_inter = max(det_bbox[0], X1)
                        y1_inter = max(det_bbox[1], Y1)
                        x2_inter = min(det_bbox[2], X2)
                        y2_inter = min(det_bbox[3], Y2)
                        
                        if x1_inter < x2_inter and y1_inter < y2_inter:
                            inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                            bbox_area = (X2 - X1) * (Y2 - Y1)
                            iou = inter_area / bbox_area
                            
                            if iou > 0.3:  # IoU > 0.3이면 같은 차량으로 판단
                                detected = True
                                conf[0][seeds] = det_bbox[4]
                                break
                
                if not detected:
                    print(f'  적외선 공격 성공! (step={steps}, seed={seeds})')
                    G_best[0] = population[seeds]
                    tag_break = 1
                    tag_run_vis = 1
                    break
                
                if conf[0][seeds] < conf_p[0][seeds]:
                    P_best[seeds] = population[seeds]
                    conf_p[0][seeds] = conf[0][seeds]
                
                if conf[0][seeds] < conf_G:
                    G_best[0] = population[seeds]
                    conf_G = conf[0][seeds]
            
            # PSO 업데이트
            for seeds in range(seed):
                for i in range(Gein_inf):
                    V[seeds][i] = omega * V[seeds][i] + \
                                 c1 * r1 * (P_best[seeds][i] - population[seeds][i]) + \
                                 c2 * r2 * (G_best[0][i] - population[seeds][i])
                    population[seeds][i] = population[seeds][i] + int(V[seeds][i])
            
            population = clip(population, X1, Y1, X2, Y2)
        
        if tag_run_vis == 0:
            print(f'  적외선 공격 실패')
            continue
        
        # 가시광선 공격
        population_color = np.zeros((seed, 18, 18, 3))
        population_color = initiation_color_18(population_color)
        
        conf_color = np.zeros((1, seed))
        P_best_color = np.zeros((seed, 18, 18, 3))
        conf_p_color = np.ones((1, seed)) * 100
        G_best_color = np.zeros((1, 18, 18, 3))
        conf_G_color = 100
        V_color = np.zeros((seed, 18, 18, 3))
        
        tag_break = 0
        for steps in range(step):
            if tag_break == 1:
                break
                
            for seeds in range(seed):
                Query[net_id] = Query[net_id] + 1
                
                unit_color = population_color[seeds]
                img_vis = cv2.imread(img_vis_path)
                
                octagon_vis1(img_vis, G_best, unit_color, X1, Y1, X2, Y2, path_adv_vis, D)
                res_vis = detector_vis(path_adv_vis)
                
                # 이 특정 차량이 탐지되지 않았는지 확인
                detected = False
                if len(res_vis[0][car_class_idx]) > 0:
                    for det_bbox in res_vis[0][car_class_idx]:
                        x1_inter = max(det_bbox[0], X1)
                        y1_inter = max(det_bbox[1], Y1)
                        x2_inter = min(det_bbox[2], X2)
                        y2_inter = min(det_bbox[3], Y2)
                        
                        if x1_inter < x2_inter and y1_inter < y2_inter:
                            inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                            bbox_area = (X2 - X1) * (Y2 - Y1)
                            iou = inter_area / bbox_area
                            
                            if iou > 0.3:
                                detected = True
                                conf_color[0][seeds] = det_bbox[4]
                                break
                
                if not detected:
                    print(f'  가시광선 공격 성공! (step={steps}, seed={seeds})')
                    
                    os.makedirs('results', exist_ok=True)
                    path_save_vis = f'results/{pic[:-4]}_vehicle{pair_idx}_vis.jpg'
                    path_save_inf = f'results/{pic[:-4]}_vehicle{pair_idx}_inf.jpg'
                    
                    cv2.imwrite(path_save_vis, cv2.imread(path_adv_vis))
                    cv2.imwrite(path_save_inf, cv2.imread(path_adv_inf))
                    
                    tag_break = 1
                    ASR[net_id] = ASR[net_id] + 1
                    attacked_vehicles += 1
                    break
                
                if conf_color[0][seeds] < conf_p_color[0][seeds]:
                    P_best_color[seeds] = population_color[seeds]
                    conf_p_color[0][seeds] = conf_color[0][seeds]
                
                if conf_color[0][seeds] < conf_G_color:
                    G_best_color[0] = population_color[seeds]
                    conf_G_color = conf_color[0][seeds]
            
            # PSO 업데이트 (컬러)
            for seeds in range(seed):
                for i in range(18):
                    for j in range(18):
                        for k in range(3):
                            V_color[seeds][i][j][k] = omega * V_color[seeds][i][j][k] + \
                                                     c1 * r1 * (P_best_color[seeds][i][j][k] - population_color[seeds][i][j][k]) + \
                                                     c2 * r2 * (G_best_color[0][i][j][k] - population_color[seeds][i][j][k])
                            population_color[seeds][i][j][k] = population_color[seeds][i][j][k] + int(V_color[seeds][i][j][k])
            
            for seeds in range(seed):
                for i in range(18):
                    for j in range(18):
                        for k in range(3):
                            if population_color[seeds][i][j][k] < 0 or population_color[seeds][i][j][k] > 255:
                                population_color[seeds][i][j][k] = random.randint(0, 255)
        
        if tag_break == 0:
            print(f'  가시광선 공격 실패')

print(f'\n{"="*60}')
print(f'최종 결과:')
print(f'전체 차량: {total_vehicles}대')
print(f'공격 성공: {attacked_vehicles}대')
print(f'ASR = {ASR}')
print(f'Query = {Query}')
print(f'count_all = {count_all}')

if total_vehicles > 0:
    print(f'성공률 = {attacked_vehicles / total_vehicles:.2%}')
if count_all[0] > 0:
    print(f'평균 Query = {Query[0] / count_all[0]:.1f}')