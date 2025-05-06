import os
import cv2

# 数据集路径
dataset_root = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX"
image_dir = os.path.join(dataset_root, "data", "images", "test")  # 测试集图像目录
label_dir = os.path.join(dataset_root, "data", "labels", "test")  # 测试集标签目录
output_dir = os.path.join(dataset_root, "visualized_labels")

# 类别名称
class_names = [
    "boneanomaly", "bonelesion", "foreignbody", "fracture",
    "metal", "periostealreaction", "pronatorsign", "softtissue", "text"
]

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 指定标签文件
label_file = "0015_0668695173_01_WRI-L1_F008.txt"  # 标签文件名
img_name = label_file.replace('.txt', '.png')  # 对应的图像文件名
img_path = os.path.join(image_dir, img_name)
label_path = os.path.join(label_dir, label_file)

# 读取图像
if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
    exit()
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Failed to read image: {img_path}")
    exit()

# 转换为 3 通道以绘制彩色框
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
h, w = img.shape
print(f"Image dimensions: {w}x{h}")

# 读取标签
if not os.path.exists(label_path):
    print(f"Label file not found: {label_path}")
    exit()
with open(label_path, 'r') as f:
    lines = f.readlines()

# 绘制目标框
for line in lines:
    if not line.strip():
        continue
    try:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        class_id = int(class_id)
        # 反归一化
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        # 计算边界框
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        # 绘制框
        color = (0, 255, 0) if class_id == 3 else (255, 0, 0)  # fracture 绿色，text 红色
        cv2.rectangle(img_color, (x1, y1), (x2, y2), color, 2)
        # 添加类别标签
        label = class_names[class_id]
        cv2.putText(img_color, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"Class: {label}, Box: ({x1}, {y1}) to ({x2}, {y2})")
    except Exception as e:
        print(f"Error processing {label_file}: {e}")
        continue

# 保存可视化结果
output_path = os.path.join(output_dir, img_name)
cv2.imwrite(output_path, img_color)
print(f"Saved visualized image: {output_path}")

# 可选：显示图像
cv2.imshow("Visualized Image", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()