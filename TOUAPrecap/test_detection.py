from detect_single_image import yolov3_inf
import cv2

img_path = 'dataset_attack/inf_0/FLIR_08864_PreviewData.jpeg'
result = yolov3_inf(img_path)

print("Result structure:")
print(f"Type: {type(result)}")
print(f"Length: {len(result)}")
for i, r in enumerate(result):
    print(f"result[{i}] shape: {r.shape if hasattr(r, 'shape') else len(r)}")
    if hasattr(r, 'shape') and r.shape[0] > 0:
        print(f"  First detection: {r[0]}")
