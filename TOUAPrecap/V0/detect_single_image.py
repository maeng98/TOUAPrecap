import os

from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import numpy as np



config_car_yolov3 = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\yolov3.py'
checkpoint_car_inf_yolov3 = r'C:\Users\a\Desktop\mmdetection_model\car_infrared\yolov3_92.1.pth'
# build the model from a config file and a checkpoint file
model_car_inf_yolov3 = init_detector(config_car_yolov3, checkpoint_car_inf_yolov3, device='cuda:0')

config_car_detr = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\detr.py'
checkpoint_car_inf_detr = r'C:\Users\a\Desktop\mmdetection_model\car_infrared\detr_94_6.pth'
# build the model from a config file and a checkpoint file
model_car_inf_detr = init_detector(config_car_detr, checkpoint_car_inf_detr, device='cuda:0')

config_car_mask = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\mask.py'
checkpoint_car_inf_mask = r'C:\Users\a\Desktop\mmdetection_model\car_infrared\mask_94_2.pth'
# build the model from a config file and a checkpoint file
model_car_inf_mask = init_detector(config_car_mask, checkpoint_car_inf_mask, device='cuda:0')

config_car_faster = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\faster.py'
checkpoint_car_inf_faster = r'C:\Users\a\Desktop\mmdetection_model\car_infrared\faster_rcnn_94_4.pth'
# build the model from a config file and a checkpoint file
model_car_inf_faster = init_detector(config_car_faster, checkpoint_car_inf_faster, device='cuda:0')

config_car_libra = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\libra.py'
checkpoint_car_inf_libra = r'C:\Users\a\Desktop\mmdetection_model\car_infrared\libra_95_6.pth'
# build the model from a config file and a checkpoint file
model_car_inf_libra = init_detector(config_car_libra, checkpoint_car_inf_libra, device='cuda:0')

config_car_retina = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\retina.py'
checkpoint_car_inf_retina = r'C:\Users\a\Desktop\mmdetection_model\car_infrared\retina_95_5.pth'
# build the model from a config file and a checkpoint file
model_car_inf_retina = init_detector(config_car_retina, checkpoint_car_inf_retina, device='cuda:0')


# config_people_yolov3 = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\yolov3.py'
# checkpoint_people_inf_yolov3 = r'C:\Users\a\PycharmProjects\pythonProject\object_detect_models\mmdetection-master\work_dirs\my_custom_dataset\yolov3_909.pth'
# # build the model from a config file and a checkpoint file
# model_people_inf_yolov3 = init_detector(config_people_yolov3, checkpoint_people_inf_yolov3, device='cuda:0')




config_car_yolov3 = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\yolov3.py'
checkpoint_car_vis_yolov3 = r'C:\Users\a\Desktop\mmdetection_model\car_visible\yolov3_816.pth'
# build the model from a config file and a checkpoint file
model_car_vis_yolov3 = init_detector(config_car_yolov3, checkpoint_car_vis_yolov3, device='cuda:0')

# config_car_yolov3 = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\yolov3_pre.py'
# checkpoint_pretrain_yolov3 = r'C:\Users\a\PycharmProjects\pythonProject\object_detect_models\mmdetection-master\work_dirs\my_custom_dataset\pre_train\yolov3_d53_320_273e_coco-421362b6.pth'
# # build the model from a config file and a checkpoint file
# model_pretrain_yolov3 = init_detector(config_car_yolov3, checkpoint_car_vis_yolov3, device='cuda:0')
#
config_car_detr = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\detr.py'
checkpoint_car_vis_detr = r'C:\Users\a\Desktop\mmdetection_model\car_visible\detr_870.pth'
# build the model from a config file and a checkpoint file
model_car_vis_detr = init_detector(config_car_detr, checkpoint_car_vis_detr, device='cuda:0')

# config_car_detr = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\detr.py'
# checkpoint_pretrain_detr = r'C:\Users\a\PycharmProjects\pythonProject\object_detect_models\mmdetection-master\work_dirs\my_custom_dataset\pre_train\detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
# # build the model from a config file and a checkpoint file
# model_pretrain_detr = init_detector(config_car_detr, checkpoint_pretrain_detr, device='cuda:0')

#
config_car_mask = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\mask.py'
checkpoint_car_vis_mask = r'C:\Users\a\Desktop\mmdetection_model\car_visible\mask_737.pth'
# build the model from a config file and a checkpoint file
model_car_vis_mask = init_detector(config_car_mask, checkpoint_car_vis_mask, device='cuda:0')

config_car_faster = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\faster.py'
checkpoint_car_vis_faster = r'C:\Users\a\Desktop\mmdetection_model\car_visible\faster_734.pth'
# build the model from a config file and a checkpoint file
model_car_vis_faster = init_detector(config_car_faster, checkpoint_car_vis_faster, device='cuda:0')

config_car_libra = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\libra.py'
checkpoint_car_vis_libra = r'C:\Users\a\Desktop\mmdetection_model\car_visible\libra_874.pth'
# build the model from a config file and a checkpoint file
model_car_vis_libra = init_detector(config_car_libra, checkpoint_car_vis_libra, device='cuda:0')

config_car_retina = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\retina.py'
checkpoint_car_vis_retina = r'C:\Users\a\Desktop\mmdetection_model\car_visible\retina_755.pth'
# build the model from a config file and a checkpoint file
model_car_vis_retina = init_detector(config_car_retina, checkpoint_car_vis_retina, device='cuda:0')


# config_people_yolov3 = r'C:\Users\a\PycharmProjects\pythonProject\cross_modal_attack\configs\yolov3.py'
# checkpoint_people_vis_yolov3 = r'C:\Users\a\PycharmProjects\pythonProject\object_detect_models\mmdetection-master\work_dirs\my_custom_dataset\pre_train\yolov3_d53_320_273e_coco-421362b6.pth'
# # build the model from a config file and a checkpoint file
# model_people_vis_yolov3 = init_detector(config_people_yolov3, checkpoint_people_vis_yolov3, device='cuda:0')




def detection(img, model):
    result = inference_detector(model, img)
    # print(result)

    score_thres = 0.5

    if len(result) == 2:
        result = list(result)
        for i in range(len(result[0])):
            if result[0][i].size != 0:
                bboxes = np.vstack(result[0][i])
                scores = bboxes[:, -1]
                inds = scores > score_thres
                bboxes = bboxes[inds, :]
                segms = result[1][i]
                filtered_segms = []
                for j, flag in enumerate(inds):
                    if flag:
                        filtered_segms.append(segms[j])
                segms = filtered_segms
                result[0][i] = bboxes
                result[1][i] = segms
        result = tuple(result)
    else:
        for i in range(len(result)):
            if result[i].size != 0:
                bboxes = np.vstack(result[i])
                scores = bboxes[:, -1]
                inds = scores > score_thres
                bboxes = bboxes[inds, :]
                result[i] = bboxes

    model.show_result(img, result, out_file='result.jpg')

    return result

def detection1(img, model):
    result = inference_detector(model, img)
    # print(result)

    score_thres = 0.5

    if len(result) == 2:
        result = list(result)
        for i in range(len(result[0])):
            if result[0][i].size != 0:
                bboxes = np.vstack(result[0][i])
                scores = bboxes[:, -1]
                inds = scores > score_thres
                bboxes = bboxes[inds, :]
                segms = result[1][i]
                filtered_segms = []
                for j, flag in enumerate(inds):
                    if flag:
                        filtered_segms.append(segms[j])
                segms = filtered_segms
                result[0][i] = bboxes
                result[1][i] = segms
        result = tuple(result)
    else:
        for i in range(len(result)):
            if result[i].size != 0:
                bboxes = np.vstack(result[i])
                scores = bboxes[:, -1]
                inds = scores > score_thres
                bboxes = bboxes[inds, :]
                result[i] = bboxes

    model.show_result(img, result, out_file='result1.jpg')

    return result

def yolov3_inf(img):
    result = detection(img, model_car_inf_yolov3)
    return result

def yolov3_inf1(img):
    result = detection1(img, model_car_inf_yolov3)
    return result

def detr_inf(img):
    result = detection(img, model_car_inf_detr)
    return result

def mask_inf(img):
    result = detection(img, model_car_inf_mask)
    return result

def faster_inf(img):
    result = detection(img, model_car_inf_faster)
    return result

def libra_inf(img):
    result = detection(img, model_car_inf_libra)
    return result

def retina_inf(img):
    result = detection(img, model_car_inf_retina)
    return result


def yolov3_vis(img):
    result = detection(img, model_car_vis_yolov3)
    return result

def yolov3_vis1(img):
    result = detection1(img, model_car_vis_yolov3)
    return result

# def yolov3_pretrain(img):
#     result = detection(img, model_pretrain_yolov3)
#     return result
#
def detr_vis(img):
    result = detection(img, model_car_vis_detr)
    return result
#
# def detr_pretrtain(img):
#     result = detection(img, model_pretrain_detr)
#     return result

#
def mask_vis(img):
    result = detection(img, model_car_vis_mask)
    return result

def faster_vis(img):
    result = detection(img, model_car_vis_faster)
    return result

def libra_vis(img):
    result = detection(img, model_car_vis_libra)
    return result

def retina_vis(img):
    result = detection(img, model_car_vis_retina)
    return result
















