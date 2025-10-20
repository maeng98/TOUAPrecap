"""
EOT를 포함한 TOUAP 구현 (논문 완전 재현)
"""
import matplotlib.pyplot as plt
import os
from detect_single_image_yolov5 import yolov5_vis, yolov5_inf
import random
import cv2
import numpy as np
from functions import initiation_8, octagon_inf, clip, initiation_color_18, octagon_vis1
from eot_transforms import apply_eot_transforms, simple_eot_transform

current_dir = os.path.dirname(os.path.abspath(__file__))

ASR = [0]
Query = [0]
count_all = [0]

detector_inf = yolov5_inf
detector_vis = yolov5_vis

net_id = 0

# 데이터셋 경로
dir_inf = os.path.join(current_dir, 'datasets/test/infrared/')
dir_vis = os.path.join(current_dir, 'datasets/test/visible/')
results_dir = os.path.join(current_dir, 'results/')
os.makedirs(results_dir, exist_ok=True)

# EOT 설정
USE_EOT = True  # EOT 사용 여부
EOT_SAMPLES = 3  # EOT 샘플 개수 (논문에서는 명시 안 함, 보통 3-5개)

file = os.listdir(dir_inf) if os.path.exists(dir_inf) else []
tag = 0

if not file:
    print(f'오류: {dir_inf} 폴더가 비어있거나 존재하지 않습니다.')
    exit(1)

for pic in file:
    tag = tag + 1
    
    img_inf_path = dir_inf + pic
    img_vis_path = dir_vis + pic
    
    path_adv_vis = 'adv_vis_dual.jpg'
    path_adv_inf = 'adv_inf_dual.jpg'
    
    res_inf1 = detector_inf(img_inf_path)
    res_vis1 = detector_vis(img_vis_path)
    
    car_class_idx = 2
    
    shape_inf = len(res_inf1[0][car_class_idx]) if len(res_inf1[0][car_class_idx]) > 0 else 0
    shape_vis = len(res_vis1[0][car_class_idx]) if len(res_vis1[0][car_class_idx]) > 0 else 0
    
    print(f'pic_id={tag}, shape_inf={shape_inf}, shape_vis={shape_vis}')
    print(f'ASR={ASR}, Query={Query}, count_all={count_all}')
    
    if shape_vis != 1 or shape_inf != 1:
        continue
    
    bbox_inf = res_inf1[0][car_class_idx][0]
    bbox_vis = res_vis1[0][car_class_idx][0]
    
    if bbox_vis[0] > (bbox_inf[0] + ((bbox_inf[2] - bbox_inf[0]) / 2)) or \
       bbox_inf[0] > (bbox_vis[0] + ((bbox_vis[2] - bbox_vis[0]) / 2)) or \
       bbox_vis[1] > (bbox_inf[1] + ((bbox_inf[3] - bbox_inf[1]) / 2)) or \
       bbox_inf[1] > (bbox_vis[1] + ((bbox_vis[3] - bbox_vis[1]) / 2)):
        continue
    
    count_all[net_id] = count_all[net_id] + 1
    
    # PSO 파라미터
    D = 18
    omega, c1, r1, c2, r2 = 0.9, 1.6, 0.5, 1.4, 0.5
    seed = 100
    step = 10
    shape = 8
    Gein_inf = 2 * shape
    
    population = np.zeros((seed, Gein_inf))
    
    X1, Y1, X2, Y2 = int(bbox_inf[0]), int(bbox_inf[1]), int(bbox_inf[2]), int(bbox_inf[3])
    
    population = initiation_8(population, X1, Y1, X2, Y2)
    conf = np.zeros((1, seed))
    P_best = np.zeros((seed, Gein_inf))
    conf_p = np.ones((1, seed)) * 100
    G_best = np.zeros((1, Gein_inf))
    conf_G = 100
    V = np.zeros((seed, Gein_inf))
    
    print(f'X1={X1}, Y1={Y1}, X2={X2}, Y2={Y2}')
    
    tag_break = 0
    tag_run_vis = 0
    
    # ============================================
    # Stage 1: 적외선 공격 (EOT 적용)
    # ============================================
    for steps in range(step):
        if tag_break == 1:
            break
            
        for seeds in range(seed):
            Query[net_id] = Query[net_id] + 1
            
            if Query[net_id] % 100 == 0:
                print(f'적외선 공격: pic={tag}, step={steps}, seed={seeds}, Query={Query[net_id]}')
            
            unit = population[seeds]
            
            if USE_EOT:
                # ===== EOT 적용 =====
                # 여러 변환에 대한 평균 confidence 계산
                eot_confidences = []
                
                for eot_sample in range(EOT_SAMPLES):
                    # 원본 이미지 읽기
                    img_inf = cv2.imread(img_inf_path)
                    
                    # 패치 적용
                    octagon_inf(img_inf, unit, path_adv_inf)
                    
                    # EOT 변환 적용
                    adv_img = cv2.imread(path_adv_inf)
                    transformed_img = simple_eot_transform(adv_img)
                    temp_path = f'temp_eot_{eot_sample}.jpg'
                    cv2.imwrite(temp_path, transformed_img)
                    
                    # 탐지
                    res_inf = detector_inf(temp_path)
                    
                    # confidence 저장
                    if len(res_inf[0][car_class_idx]) == 0:
                        eot_confidences.append(0.0)
                    else:
                        eot_confidences.append(res_inf[0][car_class_idx][0][4])
                
                # 평균 confidence 계산
                conf[0][seeds] = np.mean(eot_confidences)
                
                # 모든 변환에서 탐지 실패했는지 확인
                if all(c == 0.0 for c in eot_confidences):
                    print(f'  적외선 공격 성공! (EOT 사용, step={steps}, seed={seeds})')
                    G_best[0] = population[seeds]
                    tag_break = 1
                    tag_run_vis = 1
                    break
            
            else:
                # ===== EOT 없이 (기존 방식) =====
                img_inf = cv2.imread(img_inf_path)
                octagon_inf(img_inf, unit, path_adv_inf)
                res_inf = detector_inf(path_adv_inf)
                
                if len(res_inf[0][car_class_idx]) == 0:
                    G_best[0] = population[seeds]
                    tag_break = 1
                    tag_run_vis = 1
                    break
                
                conf[0][seeds] = res_inf[0][car_class_idx][0][4]
            
            # PSO 업데이트
            if conf[0][seeds] < conf_p[0][seeds]:
                P_best[seeds] = population[seeds]
                conf_p[0][seeds] = conf[0][seeds]
            
            if conf[0][seeds] < conf_G:
                G_best[0] = population[seeds]
                conf_G = conf[0][seeds]
        
        # PSO 위치/속도 업데이트
        for seeds in range(0, seed):
            for i in range(0, Gein_inf):
                V[seeds][i] = omega * V[seeds][i] + \
                             c1 * r1 * (P_best[seeds][i] - population[seeds][i]) + \
                             c2 * r2 * (G_best[0][i] - population[seeds][i])
                population[seeds][i] = population[seeds][i] + int(V[seeds][i])
        
        population = clip(population, X1, Y1, X2, Y2)
    
    if tag_run_vis == 0:
        print(f'  적외선 공격 실패, 다음 이미지로')
        continue
    
    img_save_inf = cv2.imread('result.jpg')
    
    # ============================================
    # Stage 2: 가시광선 공격 (EOT 적용)
    # ============================================
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
            
            if Query[net_id] % 100 == 0:
                print(f'가시광선 공격: pic={tag}, step={steps}, seed={seeds}, Query={Query[net_id]}')
            
            unit_color = population_color[seeds]
            
            if USE_EOT:
                # ===== EOT 적용 =====
                eot_confidences = []
                
                for eot_sample in range(EOT_SAMPLES):
                    img_vis = cv2.imread(img_vis_path)
                    octagon_vis1(img_vis, G_best, unit_color, X1, Y1, X2, Y2, path_adv_vis, D)
                    
                    # EOT 변환
                    adv_img = cv2.imread(path_adv_vis)
                    transformed_img = simple_eot_transform(adv_img)
                    temp_path = f'temp_eot_vis_{eot_sample}.jpg'
                    cv2.imwrite(temp_path, transformed_img)
                    
                    # 탐지
                    res_vis = detector_vis(temp_path)
                    
                    if len(res_vis[0][car_class_idx]) == 0:
                        eot_confidences.append(0.0)
                    else:
                        eot_confidences.append(res_vis[0][car_class_idx][0][4])
                
                conf_color[0][seeds] = np.mean(eot_confidences)
                
                if all(c == 0.0 for c in eot_confidences):
                    print(f'  가시광선 공격 성공! (EOT 사용, step={steps}, seed={seeds})')
                    
                    path_save_vis = os.path.join(results_dir, f'vis_{pic}')
                    path_save_inf = os.path.join(results_dir, f'inf_{pic}')
                    
                    img_save_vis = cv2.imread('result.jpg')
                    cv2.imwrite(path_save_vis, img_save_vis)
                    cv2.imwrite(path_save_inf, img_save_inf)
                    
                    tag_break = 1
                    ASR[net_id] = ASR[net_id] + 1
                    break
            
            else:
                # ===== EOT 없이 =====
                img_vis = cv2.imread(img_vis_path)
                octagon_vis1(img_vis, G_best, unit_color, X1, Y1, X2, Y2, path_adv_vis, D)
                res_vis = detector_vis(path_adv_vis)
                
                if len(res_vis[0][car_class_idx]) == 0:
                    path_save_vis = os.path.join(results_dir, f'vis_{pic}')
                    path_save_inf = os.path.join(results_dir, f'inf_{pic}')
                    
                    img_save_vis = cv2.imread('result.jpg')
                    cv2.imwrite(path_save_vis, img_save_vis)
                    cv2.imwrite(path_save_inf, img_save_inf)
                    
                    tag_break = 1
                    ASR[net_id] = ASR[net_id] + 1
                    break
                
                conf_color[0][seeds] = res_vis[0][car_class_idx][0][4]
            
            if conf_color[0][seeds] < conf_p_color[0][seeds]:
                P_best_color[seeds] = population_color[seeds]
                conf_p_color[0][seeds] = conf_color[0][seeds]
            
            if conf_color[0][seeds] < conf_G_color:
                G_best_color[0] = population_color[seeds]
                conf_G_color = conf_color[0][seeds]
        
        # PSO 업데이트 (컬러)
        for seeds in range(0, seed):
            for i in range(0, 18):
                for j in range(0, 18):
                    for k in range(0, 3):
                        V_color[seeds][i][j][k] = omega * V_color[seeds][i][j][k] + \
                                                 c1 * r1 * (P_best_color[seeds][i][j][k] - population_color[seeds][i][j][k]) + \
                                                 c2 * r2 * (G_best_color[0][i][j][k] - population_color[seeds][i][j][k])
                        population_color[seeds][i][j][k] = population_color[seeds][i][j][k] + int(V_color[seeds][i][j][k])
        
        for seeds in range(0, seed):
            for i in range(0, 18):
                for j in range(0, 18):
                    for k in range(0, 3):
                        if population_color[seeds][i][j][k] < 0 or population_color[seeds][i][j][k] > 255:
                            population_color[seeds][i][j][k] = random.randint(0, 255)

print(f'\n최종 결과 (EOT {"사용" if USE_EOT else "미사용"}):')
print(f'ASR = {ASR}')
print(f'Query = {Query}')
print(f'count_all = {count_all}')

if count_all[0] > 0:
    print(f'ASR 비율 = {ASR[0] / count_all[0]:.2%}')
    print(f'평균 Query = {Query[0] / count_all[0]:.1f}')