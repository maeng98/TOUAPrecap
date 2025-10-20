from detect_single_image import yolov3_inf, yolov3_vis

img_inf = 'dataset_attack/inf_0/FLIR_08864_PreviewData.jpeg'
img_vis = 'dataset_attack/vis_0/FLIR_08864_PreviewData.jpeg'

print("=== Infrared Detection ===")
res_inf = yolov3_inf(img_inf)
print(f"result[0] shape: {res_inf[0].shape}")
if res_inf[0].shape[0] > 0:
    print(f"First detection: {res_inf[0][0]}")
else:
    print("No detection")

print("\n=== Visible Detection ===")
res_vis = yolov3_vis(img_vis)
print(f"result[0] shape: {res_vis[0].shape}")
if res_vis[0].shape[0] > 0:
    print(f"First detection: {res_vis[0][0]}")
else:
    print("No detection")
