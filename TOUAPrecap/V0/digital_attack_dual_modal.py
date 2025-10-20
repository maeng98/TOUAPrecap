import matplotlib.pyplot as plt
import os
from detect_single_image import yolov3_vis, yolov3_inf, detr_vis, detr_inf, mask_inf, mask_vis, faster_inf, faster_vis, libra_inf, libra_vis, retina_inf, retina_vis
import random
import cv2
import numpy as np
from functions import initiation_8, octagon_inf, clip, initiation_color_18, octagon_vis1




# dir_inf = r'C:\Users\a\Desktop\LLVIP\LLVIP\infrared\test' + '/'
# dir_vis = r'C:\Users\a\Desktop\LLVIP\LLVIP\visible\test' + '/'

ASR = [0, 0, 0, 0, 0, 0]
Query = [0, 0, 0, 0, 0, 0]
count_all = [0, 0, 0, 0, 0, 0]

detector_inf = yolov3_inf
detector_vis = yolov3_vis

for net_id in range(2, 3):

    if net_id == 0:
        detector_inf = yolov3_inf
        detector_vis = yolov3_vis
    if net_id == 1:
        detector_inf = detr_inf
        detector_vis = detr_vis
    if net_id == 2:
        detector_inf = mask_inf
        detector_vis = mask_vis
    if net_id == 3:
        detector_inf = faster_inf
        detector_vis = faster_vis
    if net_id == 4:
        detector_inf = libra_inf
        detector_vis = libra_vis
    if net_id == 5:
        detector_inf = retina_inf
        detector_vis = retina_vis

    dir_inf = r'dataset_attack/inf_' + str(net_id) + '/'
    dir_vis = r'dataset_attack/vis_' + str(net_id) + '/'


    # if net_id == 2:
    #     continue









    file = os.listdir(dir_inf)
    tag = 0
    for pic in file:
        tag = tag + 1

        # if tag < 58:
        #     continue

        img_inf_path = dir_inf + pic
        img_vis_path = dir_vis + pic

        # img_inf_path = r'C:\Users\a\Desktop\LLVIP\LLVIP\infrared\test\190002.jpg'
        # img_vis_path = r'C:\Users\a\Desktop\LLVIP\LLVIP\visible\test\190002.jpg'
        path_adv_vis = 'adv_vis_dual.jpg'
        path_adv_inf = 'adv_inf_dual.jpg'

        res_inf1 = detector_inf(img_inf_path)
        res_vis1 = detector_vis(img_vis_path)

        shape_inf, b1 = res_inf1[0][2].shape
        shape_vis, b2 = res_vis1[0][2].shape

        print('net_id, pic_id, shape_inf, shape_vis = ', net_id, tag, shape_inf, shape_vis)
        print('ASR = ', ASR)
        print('Query = ', Query)
        print('count_all = ', count_all)

        if shape_vis != 1 or shape_inf != 1:
            continue

        if res_vis1[0][2][0][0] > (res_inf1[0][2][0][0] + ((res_inf1[0][2][0][2] - res_inf1[0][2][0][0]) / 2)) or res_inf1[0][2][0][0] > (
                res_vis1[0][2][0][0] + ((res_vis1[0][2][0][2] - res_vis1[0][2][0][0]) / 2)) or res_vis1[0][2][0][1] > (
                res_inf1[0][2][0][1] + ((res_inf1[0][2][0][3] - res_inf1[0][2][0][1]) / 2)) or res_inf1[0][2][0][1] > (
                res_vis1[0][2][0][1] + ((res_vis1[0][2][0][3] - res_vis1[0][2][0][1]) / 2)):

            continue


        count_all[net_id] = count_all[net_id] + 1

        D = 18
        omega, c1, r1, c2, r2 = 0.9, 1.6, 0.5, 1.4, 0.5
        seed = 100
        step = 10
        shape = 8  # 8边形
        Gein_inf = 2 * shape

        population = np.zeros((seed, Gein_inf))
        unit = np.zeros((1, Gein_inf))

        res = detector_inf(img_inf_path)
        # print(res)
        # print(res[2])
        X1, Y1, X2, Y2 = int(res[0][2][0][0]), int(res[0][2][0][1]), int(res[0][2][0][2]), int(res[0][2][0][3])
        # tag1, tag2 = (X2 - X1) / 4, (Y2 - Y1) / 4
        # X1, Y1, X2, Y2 = int(X1 + tag1), int(Y1 + tag2), int(X2 - tag1), int(Y2 - tag2)
        population = initiation_8(population, X1, Y1, X2, Y2)
        conf = np.zeros((1, seed))
        P_best = np.zeros((seed, Gein_inf))
        conf_p = np.ones((1, seed)) * 100
        G_best = np.zeros((1, Gein_inf))
        conf_G = 100
        V = np.zeros((seed, Gein_inf))

        print('X1, Y1, X2, Y2 = ', X1, Y1, X2, Y2)
        print(population)

        a, b = population.shape

        tag_break = 0
        tag_run_vis = 0
        for steps in range(step):

            if tag_break == 1:
                break
            for seeds in range(seed):

                Query[net_id] = Query[net_id] + 1

                print('红外攻击 net_id, pic_id, steps, seeds = ', net_id, tag, steps, seeds)
                print('ASR = ', ASR)
                print('Query = ', Query)
                print('count_all = ', count_all)

                unit = population[seeds]
                # print('unit = ', unit)

                img_inf = cv2.imread(img_inf_path)

                octagon_inf(img_inf, unit, path_adv_inf)

                res_inf = detector_inf(path_adv_inf)
                print('res_inf[2].shape = ', res_inf[0][2].shape)

                if res_inf[0][2].shape == (0, 5):
                    G_best[0] = population[seeds]
                    tag_break = 1

                    img_inf_best = cv2.imread(path_adv_inf)
                    path_inf_best = 'adv_inf_best.jpg'
                    # cv2.imwrite(path_inf_best, img_inf_best)

                    tag_run_vis = 1

                    break

                conf[0][seeds] = res_inf[0][2][0][4]

                print(conf)

                if conf[0][seeds] < conf_p[0][seeds]:  # 更新P_best
                    P_best[seeds] = population[seeds]
                    conf_p[0][seeds] = conf[0][seeds]

                # print(P_best)

                if conf[0][seeds] < conf_G:  # 更新G_best
                    G_best[0] = population[seeds]
                    conf_G = conf[0][seeds]

                    img_inf_best = cv2.imread('adv_inf.jpg')
                    path_inf_best = 'adv_inf_best.jpg'

                    cv2.imwrite(path_inf_best, img_inf_best)

                # print(G_best)

            for seeds in range(0, seed):
                for i in range(0, Gein_inf):
                    V[seeds][i] = omega * V[seeds][i] + c1 * r1 * (
                                P_best[seeds][i] - population[seeds][i]) + c2 * r2 * (
                                          G_best[0][i] - population[seeds][i])
                    population[seeds][i] = population[seeds][i] + int(V[seeds][i])

            # print('V = ', V)

            # print('population = ', population)

            population = clip(population, X1, Y1, X2, Y2)

            # print('x1, x2, x3, x4 = ', X1, int(X1 + (X2 - X1) / 3), int(X1 + (X2 - X1) / 3 * 2), X2)

            # print('y1, y2, y3, y4 = ', Y1, int(Y1 + (Y2 - Y1) / 3), int(Y1 + (Y2 - Y1) / 3 * 2), Y2)

            # print('population = ', population)

        if tag_run_vis == 0:
            continue

        img_save_inf = cv2.imread('result.jpg')

        population_color = np.zeros((seed, 18, 18, 3))
        population_color = initiation_color_18(population_color)
        unit_color = np.zeros((1, 18, 18, 3))

        conf_color = np.zeros((1, seed))
        P_best_color = np.zeros((seed, 18, 18, 3))
        conf_p_color = np.ones((1, seed)) * 100
        G_best_color = np.zeros((1, 18, 18, 3))
        conf_G_color = 100
        V_color = np.zeros((seed, 18, 18, 3))

        # print('population_color = ', population_color)

        tag_break = 0
        for steps in range(step):

            if tag_break == 1:
                break

            for seeds in range(seed):

                Query[net_id] = Query[net_id] + 1

                print('可见光攻击 net_id, pic_id, steps, seeds = ', net_id, tag, steps, seeds)
                print('ASR = ', ASR)
                print('Query = ', Query)
                print('count_all = ', count_all)

                unit_color = population_color[seeds]

                # print('unit_color = ', unit_color)

                img_vis = cv2.imread(img_vis_path)
                octagon_vis1(img_vis, G_best, unit_color, X1, Y1, X2, Y2, path_adv_vis, D)

                res_vis = detector_vis(path_adv_vis)

                shape_vis_adv, b3 = res_vis[0][2].shape

                # if shape_vis < shape_vis_adv:  # 增加扰动后反而多检测出来一辆车
                #     shape_vis = shape_vis_adv

                img_save_vis = cv2.imread('result.jpg')

                print('res_vis[2].shape = ', res_vis[0][2].shape)

                # img_show = plt.imread('result.jpg')
                # plt.imshow(img_show)
                # plt.show()

                tag_save = 1
                if res_vis[0][2].shape == (0, 5):
                    path_save_vis = 'path_adv/digital_dual_1/' + 'vis_' + pic
                    path_save_inf = 'path_adv/digital_dual_1/' + 'inf_' + pic

                    # cv2.imwrite(path_save_vis, img_save_vis)
                    # cv2.imwrite(path_save_inf, img_save_inf)

                    tag_save = tag_save + 1

                    tag_break = 1

                    ASR[net_id] = ASR[net_id] + 1

                    break

                conf_color[0][seeds] = res_vis[0][2][0][4]

                if conf_color[0][seeds] < conf_p_color[0][seeds]:  # 更新P_best_color
                    P_best_color[seeds] = population_color[seeds]
                    conf_p_color[0][seeds] = conf_color[0][seeds]

                # print(P_best)

                if conf_color[0][seeds] < conf_G_color:  # 更新G_best_color
                    G_best_color[0] = population_color[seeds]
                    conf_G_color = conf_color[0][seeds]

                for seeds in range(0, seed):
                    for i in range(0, 18):
                        for j in range(0, 18):
                            for k in range(0, 3):
                                V_color[seeds][i][j][k] = omega * V_color[seeds][i][j][k] + c1 * r1 * (
                                        P_best_color[seeds][i][j][k] - population_color[seeds][i][j][k]) + c2 * r2 * (
                                                                  G_best_color[0][i][j][k] -
                                                                  population_color[seeds][i][j][k])
                                population_color[seeds][i][j][k] = population_color[seeds][i][j][k] + int(
                                    V_color[seeds][i][j][k])

                for seeds in range(0, seed):
                    for i in range(0, 18):
                        for j in range(0, 18):
                            for k in range(0, 3):
                                if population_color[seeds][i][j][k] < 0 or population_color[seeds][i][j][k] > 255:
                                    population_color[seeds][i][j][k] = random.randint(0, 255)

                # print(G_best)

        # if tag == 1:
        #     break





print('ASR = ', ASR)
print('Query = ', Query)
print('coung_all', count_all)

# for i in range(0, 6):
#     ASR[i] = ASR[i] / count_all[i]
#     Query[i] = Query[i] / count_all[i]
#
# print('ASR = ', ASR)
# print('Query = ', Query)

# print('ASR/739 = ', ASR/739)
# print('Query/739 = ', Query/739)


















