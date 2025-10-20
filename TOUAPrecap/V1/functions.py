# from detect_single_image import yolov3_vis, yolov3_inf
import random
import cv2
import numpy as np


def octagon_inf_random(img_inf, X1, Y1, X2, Y2, path_inf):
    tag = Y1
    tag1, tag2 = (X2 - X1) / 4, (Y2 - Y1) / 4
    X1, Y1, X2, Y2 = int(X1 + tag1), int(Y1 + tag2), int(X2 - tag1), int(Y2 - tag2)
    X1, X2, Y1, Y2 = int(X1), int(X2), int(Y1), int(Y2)
    if X1 < 0:
        X1 =0
    if Y1 < 0:
        Y1 =0
    x1, x7, x8 = random.randint(int(X1), int(X1 + (X2 - X1) / 3)), random.randint(int(X1), int(X1 + (X2 - X1) / 3)), random.randint(int(X1), int(X1 + (X2 - X1) / 3))
    x2, x6, x9 = random.randint(int(X1 + (X2 - X1) / 3), int(X1 + (X2 - X1) / 3 * 2)), random.randint(int(X1 + (X2 - X1) / 3), int(X1 + (X2 - X1) / 3 * 2)), random.randint(int(X1 + (X2 - X1) / 3), int(X1 + (X2 - X1) / 3 * 2))
    x3, x4, x5 = random.randint(int(X2 - (X2 - X1) / 3), X2), random.randint(int(X2 - (X2 - X1) / 3), X2), random.randint(int(X2 - (X2 - X1) / 3), X2)
    y1, y2, y3 = random.randint(int(Y1), int(Y1 + (Y2 - Y1) / 3)), random.randint(int(Y1), int(Y1 + (Y2 - Y1) / 3)), random.randint(int(Y1), int(Y1 + (Y2 - Y1) / 3)),
    y4, y9, y8 = random.randint(int(Y1 + (Y2 - Y1) / 3), int(Y1 + (Y2 - Y1) / 3 * 2)), random.randint(int(Y1 + (Y2 - Y1) / 3), int(Y1 + (Y2 - Y1) / 3 * 2)), random.randint(int(Y1 + (Y2 - Y1) / 3), int(Y1 + (Y2 - Y1) / 3 * 2))
    y5, y6, y7 = random.randint(int(Y2 - (Y2 - Y1) / 3), Y2), random.randint(int(Y2 - (Y2 - Y1) / 3), Y2), random.randint(int(Y2 - (Y2 - Y1) / 3), Y2)

    points_inf = np.array([[(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6),(x7, y7), (x8, y8)]], np.int32)
    cv2.fillPoly(img_inf, points_inf, (0, 0, 0))

    cv2.imwrite(path_inf, img_inf)

    return x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8

def octagon_vis_random(img_vis, X1, Y1, X2, Y2, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, path_vis, D):


    img = np.ones((img_vis.shape[0], img_vis.shape[1], 3))  # 白色背景

    tag1 = int((X2 - X1) / D)
    if tag1 <= 0:
        tag1 = 1

    tag2 = int((Y2 - Y1) / D)
    if tag2 <= 0:
        tag2 = 1

    for i in range(X1 + 35, X2 + 35, tag1):  # 彩色二维码的位置，+35表示可见光与红外图像位置偏差
        for j in range(Y1, Y2, tag2):
            topleft, downright = (i, j), (int(i + (X2 - X1) / D), int(j + (Y2 - Y1) / D))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # print(topleft, downright, color)
            cv2.rectangle(img, topleft, downright, color, -1)  # 填充

    cv2.imwrite('WR.jpg', img)

    img_QR = cv2.imread('WR.jpg')

    mask = np.zeros(img_QR.shape[:2], dtype="uint8")

    points_vis = np.array([[(x1 + 35, y1 - 2), (x2 + 35, y2 - 2), (x3 + 35, y3 - 2), (x4 + 35, y4 - 2), (x5 + 35, y5 - 2), (x6 + 35, y6 - 2), (x7 + 35, y7 - 2), (x8 + 35, y8 - 2)]], np.int32)

    cv2.fillPoly(mask, points_vis, (255, 255, 255))

    masked = cv2.bitwise_and(img_QR, img_QR, mask=mask)

    # cv2.imshow("Rectangular Mask Applied to Image", masked)
    # cv2.waitKey(0)

    TACH_PATTERN_PATH2 = r'mask.jpg'
    cv2.imwrite(TACH_PATTERN_PATH2, masked)

    cv2.polylines(img_vis, points_vis, True, (255, 255, 255), 1)
    cv2.fillPoly(img_vis, points_vis, (0, 0, 0))  # 纯色填充
    # cv2.imwrite('C:\Users\User\Desktop\5_res.jpg', img66)

    # cv2.imshow("Rectangular Mask Applied to Image", img_bg)
    # cv2.waitKey(0)

    res = cv2.add(masked, img_vis)
    cv2.imwrite(path_vis, res)

def initiation_8(population, X1, Y1, X2, Y2):

    a, b = population.shape

    # print('a, b = ', a, b)

    for i in range(0, a):
        population[i][0], population[i][1] = random.randint(int(X1), int(X1 + (X2 - X1) / 3)), random.randint(int(Y1), int(Y1 + (Y2 - Y1) / 3))
        population[i][2], population[i][3] = random.randint(int(X1 + (X2 - X1) / 3), int(X1 + (X2 - X1) / 3 * 2)), random.randint(int(Y1), int(Y1 + (Y2 - Y1) / 3))
        population[i][4], population[i][5] = random.randint(int(X2 - (X2 - X1) / 3), X2), random.randint(int(Y1), int(Y1 + (Y2 - Y1) / 3))
        population[i][6], population[i][7] = random.randint(int(X2 - (X2 - X1) / 3), X2), random.randint(int(Y1 + (Y2 - Y1) / 3), int(Y1 + (Y2 - Y1) / 3 * 2))
        population[i][8], population[i][9] = random.randint(int(X2 - (X2 - X1) / 3), X2), random.randint(int(Y2 - (Y2 - Y1) / 3), Y2)
        population[i][10], population[i][11] = random.randint(int(X1 + (X2 - X1) / 3), int(X1 + (X2 - X1) / 3 * 2)), random.randint(int(Y2 - (Y2 - Y1) / 3), Y2)
        population[i][12], population[i][13] = random.randint(int(X1), int(X1 + (X2 - X1) / 3)), random.randint(int(Y2 - (Y2 - Y1) / 3), Y2)
        population[i][14], population[i][15] = random.randint(int(X1), int(X1 + (X2 - X1) / 3)), random.randint(int(Y1 + (Y2 - Y1) / 3), int(Y1 + (Y2 - Y1) / 3 * 2))

    return population


def initiation_color_18(population_color):

    a, b, c, d = population_color.shape

    for i in range(0, a):
        for j in range(0, b):
            for k in range(0, c):
                for l in range(0, d):
                    population_color[i][j][k][l] = random.randint(0, 255)



    return population_color

# P = np.zeros((18, 18, 3))
#
# P = initiation_color_18(P)
#
# print(P)



def octagon_inf(img_inf, unit, path_inf):



    points_inf = np.array([[(unit[0], unit[1]), (unit[2], unit[3]), (unit[4], unit[5]), (unit[6], unit[7]), (unit[8], unit[9]), (unit[10], unit[11]),(unit[12], unit[13]), (unit[14], unit[15])]], np.int32)
    cv2.fillPoly(img_inf, points_inf, (0, 0, 0))

    cv2.imwrite(path_inf, img_inf)




def octagon_inf2(img_inf, unit, path_inf):


    img = np.ones((img_inf.shape[0], img_inf.shape[1], 3))  # 白色背景


    points_inf = np.array([[(unit[0], unit[1]), (unit[2], unit[3]), (unit[4], unit[5]), (unit[6], unit[7]), (unit[8], unit[9]), (unit[10], unit[11]),(unit[12], unit[13]), (unit[14], unit[15])]], np.int32)
    cv2.fillPoly(img_inf, points_inf, (0, 0, 0))

    cv2.imwrite(path_inf, img_inf)

    cv2.fillPoly(img, points_inf, (10, 10, 10))

    cv2.imwrite('mask_inf.jpg', img)

def octagon_vis2(img_vis, G_best, unit, X1, Y1, X2, Y2, path_vis, path_mask, D):


    # print('unit = ', unit)


    img = np.ones((img_vis.shape[0], img_vis.shape[1], 3))  # 白色背景

    tag_i = 0
    for i in range(X1+38, X2+38, int((X2 - X1) / D)+1):  # 彩色二维码的位置，+35表示可见光与红外图像位置偏差
        tag_i = tag_i + 1
        tag_j = 0
        for j in range(Y1, Y2, int((Y2 - Y1) / D)+1):

            tag_j = tag_j + 1

            # print('Y1, Y2, j = ', Y1, Y2, j)
            # print('tag_i, tag_j = ', tag_i, tag_j)

            topleft, downright = (i, j), (int(i + (X2 - X1) / D), int(j + (Y2 - Y1) / D))
            color = (unit[tag_i-1][tag_j-1][0], unit[tag_i-1][tag_j-1][1], unit[tag_i-1][tag_j-1][2])
            # print(topleft, downright, color)
            cv2.rectangle(img, topleft, downright, color, -1)  # 填充

    cv2.imwrite('WR1.jpg', img)

    img_QR = cv2.imread('WR1.jpg')

    mask = np.zeros(img_QR.shape[:2], dtype="uint8")

    points_vis = np.array([[(G_best[0][0]+38, G_best[0][1]), (G_best[0][2]+38, G_best[0][3]), (G_best[0][4]+38, G_best[0][5]), (G_best[0][6]+38, G_best[0][7]), (G_best[0][8]+38, G_best[0][9]), (G_best[0][10]+38, G_best[0][11]), (G_best[0][12]+38, G_best[0][13]), (G_best[0][14]+38, G_best[0][15])]], np.int32)

    cv2.fillPoly(mask, points_vis, (255, 255, 255))

    masked = cv2.bitwise_and(img_QR, img_QR, mask=mask)

    # cv2.imshow("Rectangular Mask Applied to Image", masked)
    # cv2.waitKey(0)

    TACH_PATTERN_PATH2 = path_mask
    cv2.imwrite(TACH_PATTERN_PATH2, masked)

    cv2.polylines(img_vis, points_vis, True, (255, 255, 255), 1)
    cv2.fillPoly(img_vis, points_vis, (0, 0, 0))  # 纯色填充
    # cv2.imwrite('C:\Users\User\Desktop\5_res.jpg', img66)

    # cv2.imshow("Rectangular Mask Applied to Image", img_bg)
    # cv2.waitKey(0)

    res = cv2.add(masked, img_vis)
    cv2.imwrite(path_vis, res)

def octagon_vis(img_vis, G_best, unit, X1, Y1, X2, Y2, path_vis, D):


    # print('unit = ', unit)


    img = np.ones((img_vis.shape[0], img_vis.shape[1], 3))  # 白色背景

    tag_i = 0
    for i in range(X1, X2, int((X2 - X1) / D)+1):  # 彩色二维码的位置，+35表示可见光与红外图像位置偏差
        tag_i = tag_i + 1
        tag_j = 0
        for j in range(Y1, Y2, int((Y2 - Y1) / D)+1):

            tag_j = tag_j + 1

            # print('Y1, Y2, j = ', Y1, Y2, j)
            # print('tag_i, tag_j = ', tag_i, tag_j)

            topleft, downright = (i, j), (int(i + (X2 - X1) / D), int(j + (Y2 - Y1) / D))
            color = (unit[tag_i-1][tag_j-1][0], unit[tag_i-1][tag_j-1][1], unit[tag_i-1][tag_j-1][2])
            # print(topleft, downright, color)
            cv2.rectangle(img, topleft, downright, color, -1)  # 填充

    cv2.imwrite('WR.jpg', img)

    img_QR = cv2.imread('WR.jpg')

    mask = np.zeros(img_QR.shape[:2], dtype="uint8")

    points_vis = np.array([[(G_best[0][0], G_best[0][1]), (G_best[0][2], G_best[0][3]), (G_best[0][4], G_best[0][5]), (G_best[0][6], G_best[0][7]), (G_best[0][8], G_best[0][9]), (G_best[0][10], G_best[0][11]), (G_best[0][12], G_best[0][13]), (G_best[0][14], G_best[0][15])]], np.int32)

    cv2.fillPoly(mask, points_vis, (255, 255, 255))

    masked = cv2.bitwise_and(img_QR, img_QR, mask=mask)

    # cv2.imshow("Rectangular Mask Applied to Image", masked)
    # cv2.waitKey(0)

    TACH_PATTERN_PATH2 = r'mask.jpg'
    cv2.imwrite(TACH_PATTERN_PATH2, masked)

    cv2.polylines(img_vis, points_vis, True, (255, 255, 255), 1)
    cv2.fillPoly(img_vis, points_vis, (0, 0, 0))  # 纯色填充
    # cv2.imwrite('C:\Users\User\Desktop\5_res.jpg', img66)

    # cv2.imshow("Rectangular Mask Applied to Image", img_bg)
    # cv2.waitKey(0)

    res = cv2.add(masked, img_vis)
    cv2.imwrite(path_vis, res)


def octagon_vis3(img_vis, G_best, unit, X1, Y1, X2, Y2, path_vis, D):


    # print('unit = ', unit)

    x1, x7, x8 = random.randint(int(X1), int(X1 + (X2 - X1) / 3)), random.randint(int(X1), int(X1 + (
                X2 - X1) / 3)), random.randint(int(X1), int(X1 + (X2 - X1) / 3))
    x2, x6, x9 = random.randint(int(X1 + (X2 - X1) / 3), int(X1 + (X2 - X1) / 3 * 2)), random.randint(
        int(X1 + (X2 - X1) / 3), int(X1 + (X2 - X1) / 3 * 2)), random.randint(int(X1 + (X2 - X1) / 3),
                                                                              int(X1 + (X2 - X1) / 3 * 2))
    x3, x4, x5 = random.randint(int(X2 - (X2 - X1) / 3), X2), random.randint(int(X2 - (X2 - X1) / 3),
                                                                             X2), random.randint(
        int(X2 - (X2 - X1) / 3), X2)
    y1, y2, y3 = random.randint(int(Y1), int(Y1 + (Y2 - Y1) / 3)), random.randint(int(Y1), int(Y1 + (
                Y2 - Y1) / 3)), random.randint(int(Y1), int(Y1 + (Y2 - Y1) / 3)),
    y4, y9, y8 = random.randint(int(Y1 + (Y2 - Y1) / 3), int(Y1 + (Y2 - Y1) / 3 * 2)), random.randint(
        int(Y1 + (Y2 - Y1) / 3), int(Y1 + (Y2 - Y1) / 3 * 2)), random.randint(int(Y1 + (Y2 - Y1) / 3),
                                                                              int(Y1 + (Y2 - Y1) / 3 * 2))
    y5, y6, y7 = random.randint(int(Y2 - (Y2 - Y1) / 3), Y2), random.randint(int(Y2 - (Y2 - Y1) / 3),
                                                                             Y2), random.randint(
        int(Y2 - (Y2 - Y1) / 3), Y2)


    img = np.ones((img_vis.shape[0], img_vis.shape[1], 3))  # 白色背景

    tag_i = 0
    for i in range(X1, X2, int((X2 - X1) / D)+1):  # 彩色二维码的位置，+35表示可见光与红外图像位置偏差
        tag_i = tag_i + 1
        tag_j = 0
        for j in range(Y1, Y2, int((Y2 - Y1) / D)+1):

            tag_j = tag_j + 1

            # print('Y1, Y2, j = ', Y1, Y2, j)
            # print('tag_i, tag_j = ', tag_i, tag_j)

            topleft, downright = (i, j), (int(i + (X2 - X1) / D), int(j + (Y2 - Y1) / D))
            color = (unit[tag_i-1][tag_j-1][0], unit[tag_i-1][tag_j-1][1], unit[tag_i-1][tag_j-1][2])
            # print(topleft, downright, color)
            cv2.rectangle(img, topleft, downright, color, -1)  # 填充

    cv2.imwrite('WR.jpg', img)

    img_QR = cv2.imread('WR.jpg')

    mask = np.zeros(img_QR.shape[:2], dtype="uint8")

    points_vis = np.array([[(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6),(x7, y7), (x8, y8)]], np.int32)

    cv2.fillPoly(mask, points_vis, (255, 255, 255))

    masked = cv2.bitwise_and(img_QR, img_QR, mask=mask)

    # cv2.imshow("Rectangular Mask Applied to Image", masked)
    # cv2.waitKey(0)

    TACH_PATTERN_PATH2 = r'mask.jpg'
    cv2.imwrite(TACH_PATTERN_PATH2, masked)

    cv2.polylines(img_vis, points_vis, True, (255, 255, 255), 1)
    cv2.fillPoly(img_vis, points_vis, (0, 0, 0))  # 纯色填充
    # cv2.imwrite('C:\Users\User\Desktop\5_res.jpg', img66)

    # cv2.imshow("Rectangular Mask Applied to Image", img_bg)
    # cv2.waitKey(0)

    res = cv2.add(masked, img_vis)
    cv2.imwrite(path_vis, res)

def octagon_vis1(img_vis, G_best, unit, X1, Y1, X2, Y2, path_vis, D):


    # print('unit = ', unit)


    img = np.ones((img_vis.shape[0], img_vis.shape[1], 3))  # 白色背景

    tag_i = 0
    for i in range(X1, X2, int((X2 - X1) / D)+1):  # 彩色二维码的位置，+35表示可见光与红外图像位置偏差
        tag_i = tag_i + 1
        tag_j = 0
        for j in range(Y1, Y2, int((Y2 - Y1) / D)+1):

            tag_j = tag_j + 1

            # print('Y1, Y2, j = ', Y1, Y2, j)
            # print('tag_i, tag_j = ', tag_i, tag_j)

            topleft, downright = (i, j), (int(i + (X2 - X1) / D), int(j + (Y2 - Y1) / D))
            color = (unit[tag_i-1][tag_j-1][0], unit[tag_i-1][tag_j-1][1], unit[tag_i-1][tag_j-1][2])
            # print(topleft, downright, color)
            cv2.rectangle(img, topleft, downright, color, -1)  # 填充

    cv2.imwrite('WR1.jpg', img)

    img_QR = cv2.imread('WR1.jpg')

    mask = np.zeros(img_QR.shape[:2], dtype="uint8")

    points_vis = np.array([[(G_best[0][0], G_best[0][1]), (G_best[0][2], G_best[0][3]), (G_best[0][4], G_best[0][5]), (G_best[0][6], G_best[0][7]), (G_best[0][8], G_best[0][9]), (G_best[0][10], G_best[0][11]), (G_best[0][12], G_best[0][13]), (G_best[0][14], G_best[0][15])]], np.int32)

    cv2.fillPoly(mask, points_vis, (255, 255, 255))

    masked = cv2.bitwise_and(img_QR, img_QR, mask=mask)

    # cv2.imshow("Rectangular Mask Applied to Image", masked)
    # cv2.waitKey(0)

    TACH_PATTERN_PATH2 = r'mask1.jpg'
    cv2.imwrite(TACH_PATTERN_PATH2, masked)

    cv2.polylines(img_vis, points_vis, True, (255, 255, 255), 1)
    cv2.fillPoly(img_vis, points_vis, (0, 0, 0))  # 纯色填充
    # cv2.imwrite('C:\Users\User\Desktop\5_res.jpg', img66)

    # cv2.imshow("Rectangular Mask Applied to Image", img_bg)
    # cv2.waitKey(0)

    res = cv2.add(masked, img_vis)
    cv2.imwrite(path_vis, res)


def clip(population, X1, Y1, X2, Y2):

    a, b = population.shape

    # print('a, b = ', a, b)

    for i in range(0, a):

        if population[i][0] not in range(int(X1), int(X1 + (X2 - X1) / 3)):
            population[i][0] = random.randint(int(X1), int(X1 + (X2 - X1) / 3))
        if population[i][12] not in range(int(X1), int(X1 + (X2 - X1) / 3)):
            population[i][12] = random.randint(int(X1), int(X1 + (X2 - X1) / 3))
        if population[i][14] not in range(int(X1), int(X1 + (X2 - X1) / 3)):
            population[i][14] = random.randint(int(X1), int(X1 + (X2 - X1) / 3))

        if population[i][2] not in range(int(X1 + (X2 - X1) / 3), int(X1 + (X2 - X1) / 3 * 2)):
            population[i][2] = random.randint(int(X1 + (X2 - X1) / 3), int(X1 + (X2 - X1) / 3 * 2))
        if population[i][10] not in range(int(X1 + (X2 - X1) / 3), int(X1 + (X2 - X1) / 3 * 2)):
            population[i][10] = random.randint(int(X1 + (X2 - X1) / 3), int(X1 + (X2 - X1) / 3 * 2))

        if population[i][4] not in range(int(X2 - (X2 - X1) / 3), X2):
            population[i][4] = random.randint(int(X2 - (X2 - X1) / 3), X2)
        if population[i][6] not in range(int(X2 - (X2 - X1) / 3), X2):
            population[i][6] = random.randint(int(X2 - (X2 - X1) / 3), X2)
        if population[i][8] not in range(int(X2 - (X2 - X1) / 3), X2):
            population[i][8] = random.randint(int(X2 - (X2 - X1) / 3), X2)

        if population[i][1] not in range(int(Y1), int(Y1 + (Y2 - Y1) / 3)):
            population[i][1] = random.randint(int(Y1), int(Y1 + (Y2 - Y1) / 3))
        if population[i][3] not in range(int(Y1), int(Y1 + (Y2 - Y1) / 3)):
            population[i][3] = random.randint(int(Y1), int(Y1 + (Y2 - Y1) / 3))
        if population[i][5] not in range(int(Y1), int(Y1 + (Y2 - Y1) / 3)):
            population[i][5] = random.randint(int(Y1), int(Y1 + (Y2 - Y1) / 3))

        if population[i][7] not in range(int(Y1 + (Y2 - Y1) / 3), int(Y1 + (Y2 - Y1) / 3 * 2)):
            population[i][7] = random.randint(int(Y1 + (Y2 - Y1) / 3), int(Y1 + (Y2 - Y1) / 3 * 2))
        if population[i][15] not in range(int(Y1 + (Y2 - Y1) / 3), int(Y1 + (Y2 - Y1) / 3 * 2)):
            population[i][15] = random.randint(int(Y1 + (Y2 - Y1) / 3), int(Y1 + (Y2 - Y1) / 3 * 2))

        if population[i][9] not in range(int(Y2 - (Y2 - Y1) / 3), Y2):
            population[i][9] = random.randint(int(Y2 - (Y2 - Y1) / 3), Y2)
        if population[i][11] not in range(int(Y2 - (Y2 - Y1) / 3), Y2):
            population[i][11] = random.randint(int(Y2 - (Y2 - Y1) / 3), Y2)
        if population[i][13] not in range(int(Y2 - (Y2 - Y1) / 3), Y2):
            population[i][13] = random.randint(int(Y2 - (Y2 - Y1) / 3), Y2)


    return population

def octagon_vis_random1(img_vis, D):


    img = np.ones((img_vis.shape[0], img_vis.shape[1], 3))  # 白色背景

    X1, Y1, X2, Y2 = 0, 0, img_vis.shape[0], img_vis.shape[0]

    tag1 = int((X2 - X1) / D)
    if tag1 <= 0:
        tag1 = 1

    tag2 = int((Y2 - Y1) / D)
    if tag2 <= 0:
        tag2 = 1

    for i in range(X1, X2, tag1):  # 彩色二维码的位置，+35表示可见光与红外图像位置偏差
        for j in range(Y1, Y2, tag2):
            topleft, downright = (i, j), (int(i + (X2 - X1) / D), int(j + (Y2 - Y1) / D))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # print(topleft, downright, color)
            cv2.rectangle(img, topleft, downright, color, -1)  # 填充

    cv2.imwrite('WR.jpg', img)


img = cv2.imread('path_adv/clean/1_inf.jpg')

octagon_vis_random1(img, 18)






