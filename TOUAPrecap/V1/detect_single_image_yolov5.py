import torch
import cv2
import numpy as np
import sys
import os

# yolov5 폴더를 path에 추가 (상위 디렉토리)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# YOLOv5 모델 로드 (한 번만 로드)
# 경로를 yolov5 기준 상대경로로 수정
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 적외선 이미지용 모델들
model_car_inf_yolov5 = torch.hub.load('ultralytics/yolov5', 'custom', 
                                       path=os.path.join(base_path, 'runs/train/IRFLIR/weights/best.pt'),
                                       device='cuda:0')  # 또는 'cpu'

# 가시광선 이미지용 모델들  
model_car_vis_yolov5 = torch.hub.load('ultralytics/yolov5', 'custom',
                                       path=os.path.join(base_path, 'runs/train/visible_model/weights/best.pt'),
                                       device='cuda:0')  # 또는 'cpu'

# confidence threshold 설정
model_car_inf_yolov5.conf = 0.5
model_car_vis_yolov5.conf = 0.5


def detection_yolov5(img_path, model, save_path='result.jpg'):
    """
    YOLOv5로 객체 탐지 수행
    
    Args:
        img_path: 이미지 경로 또는 numpy array
        model: YOLOv5 모델
        save_path: 결과 이미지 저장 경로
    
    Returns:
        result: 탐지 결과 (MMDetection 형식과 호환되도록 변환)
    """
    # 추론 수행
    results = model(img_path)
    
    # 결과를 이미지에 그리기
    results.save(save_dir='runs/detect', exist_ok=True)
    
    # 결과를 특정 경로에 저장
    img = cv2.imread(img_path) if isinstance(img_path, str) else img_path
    rendered_img = results.render()[0]
    cv2.imwrite(save_path, rendered_img)
    
    # MMDetection 형식과 호환되도록 결과 변환
    # YOLOv5 결과: results.xyxy[0] = tensor([[x1, y1, x2, y2, conf, class], ...])
    detections = results.xyxy[0].cpu().numpy()
    
    # 클래스별로 분리 (MMDetection 형식 맞추기)
    num_classes = int(detections[:, 5].max()) + 1 if len(detections) > 0 else 1
    result = [np.array([]) for _ in range(num_classes)]
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        bbox = np.array([[x1, y1, x2, y2, conf]])
        
        if result[cls].size == 0:
            result[cls] = bbox
        else:
            result[cls] = np.vstack([result[cls], bbox])
    
    return [result]  # MMDetection과 같은 형식으로 반환


def yolov5_inf(img):
    """적외선 이미지 탐지"""
    result = detection_yolov5(img, model_car_inf_yolov5, 'result.jpg')
    return result


def yolov5_inf1(img):
    """적외선 이미지 탐지 (다른 저장 경로)"""
    result = detection_yolov5(img, model_car_inf_yolov5, 'result1.jpg')
    return result


def yolov5_vis(img):
    """가시광선 이미지 탐지"""
    result = detection_yolov5(img, model_car_vis_yolov5, 'result.jpg')
    return result


def yolov5_vis1(img):
    """가시광선 이미지 탐지 (다른 저장 경로)"""
    result = detection_yolov5(img, model_car_vis_yolov5, 'result1.jpg')
    return result


# 로컬 모델 사용 시 (이미 학습된 모델이 있는 경우)
def load_local_yolov5_model(weight_path, device='cuda:0'):
    """
    로컬에 있는 YOLOv5 모델 로드
    
    Args:
        weight_path: .pt 파일 경로
        device: 'cuda:0' 또는 'cpu'
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight_path, device=device)
    model.conf = 0.5  # confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    return model