"""
Expectation Over Transformation (EOT) for TOUAP
논문에 명시된 대로 robustness를 위한 변환 적용
"""
import cv2
import numpy as np
import random


def apply_eot_transforms(img, num_samples=5):
    """
    EOT: 여러 변환을 적용하여 물리적 robustness 향상
    
    Args:
        img: 입력 이미지
        num_samples: EOT 샘플 개수 (기본 5개)
    
    Returns:
        transformed_images: 변환된 이미지 리스트
    """
    transformed_images = []
    
    for _ in range(num_samples):
        transformed = img.copy()
        
        # 1. View Transformation (회전)
        angle = random.uniform(-15, 15)  # ±15도
        transformed = rotate_image(transformed, angle)
        
        # 2. Brightness Adjustment
        brightness_factor = random.uniform(0.7, 1.3)  # 70% ~ 130%
        transformed = adjust_brightness(transformed, brightness_factor)
        
        # 3. Contrast Adjustment
        contrast_factor = random.uniform(0.8, 1.2)
        transformed = adjust_contrast(transformed, contrast_factor)
        
        # 4. Gaussian Noise
        if random.random() > 0.5:
            noise_sigma = random.uniform(0, 10)
            transformed = add_gaussian_noise(transformed, noise_sigma)
        
        # 5. Slight Translation (약간의 이동)
        tx = random.randint(-5, 5)
        ty = random.randint(-5, 5)
        transformed = translate_image(transformed, tx, ty)
        
        # 6. Scale (약간의 확대/축소)
        scale_factor = random.uniform(0.95, 1.05)
        transformed = scale_image(transformed, scale_factor)
        
        transformed_images.append(transformed)
    
    return transformed_images


def rotate_image(img, angle):
    """이미지 회전"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), 
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def adjust_brightness(img, factor):
    """밝기 조정"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_contrast(img, factor):
    """대비 조정"""
    mean = np.mean(img)
    adjusted = (img - mean) * factor + mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def add_gaussian_noise(img, sigma):
    """가우시안 노이즈 추가"""
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def translate_image(img, tx, ty):
    """이미지 이동"""
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                borderMode=cv2.BORDER_REPLICATE)
    return translated


def scale_image(img, scale_factor):
    """이미지 확대/축소"""
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    if scale_factor > 1.0:
        # 확대 후 중앙 크롭
        scaled = cv2.resize(img, (new_w, new_h))
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        return scaled[start_y:start_y+h, start_x:start_x+w]
    else:
        # 축소 후 패딩
        scaled = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros_like(img)
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = scaled
        return canvas


def eot_loss(detector, img_path, patch_params, num_samples=5):
    """
    EOT를 사용한 loss 계산 (평균 confidence)
    
    Args:
        detector: 탐지 모델
        img_path: 원본 이미지 경로
        patch_params: 패치 파라미터
        num_samples: EOT 샘플 개수
    
    Returns:
        avg_confidence: 평균 confidence
    """
    img = cv2.imread(img_path)
    
    # 패치 적용 (octagon_inf 등 사용)
    from functions import octagon_inf
    octagon_inf(img, patch_params, 'temp_adv.jpg')
    
    # EOT 변환 적용
    adv_img = cv2.imread('temp_adv.jpg')
    transformed_images = apply_eot_transforms(adv_img, num_samples)
    
    # 각 변환된 이미지에 대해 탐지
    confidences = []
    for transformed_img in transformed_images:
        cv2.imwrite('temp_transform.jpg', transformed_img)
        result = detector('temp_transform.jpg')
        
        # confidence 추출
        if len(result[0][2]) > 0:  # 탐지된 경우
            conf = result[0][2][0][4]
            confidences.append(conf)
        else:  # 탐지 안 된 경우
            confidences.append(0.0)
    
    # 평균 confidence 반환
    avg_confidence = np.mean(confidences) if confidences else 0.0
    
    return avg_confidence


# 논문의 EOT를 위한 간단한 버전
def simple_eot_transform(img):
    """
    단순화된 EOT (빠른 실행을 위해)
    랜덤하게 1-2개의 변환만 적용
    """
    transforms = [
        lambda x: rotate_image(x, random.uniform(-10, 10)),
        lambda x: adjust_brightness(x, random.uniform(0.8, 1.2)),
        lambda x: adjust_contrast(x, random.uniform(0.9, 1.1)),
        lambda x: add_gaussian_noise(x, random.uniform(0, 5))
    ]
    
    # 랜덤하게 1-2개 선택
    selected = random.sample(transforms, random.randint(1, 2))
    
    result = img.copy()
    for transform in selected:
        result = transform(result)
    
    return result