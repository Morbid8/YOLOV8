import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# 设置数据集路径
dataset_root = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX"
image_dirs = [
    r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\gray_images\images_part1",
    r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\gray_images\images_part2",
    r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\gray_images\images_part3",
    r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\gray_images\images_part4"
]
label_dir = r"/folder_structure/yolov5/labels"  # 更新为正确的路径
output_dir = os.path.join(dataset_root, "data")
csv_path = os.path.join(dataset_root, "dataset.csv")

# 确保输入目录存在
for image_dir in image_dirs:
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
if not os.path.exists(label_dir):
    raise FileNotFoundError(f"Label directory not found: {label_dir}")

# 读取 dataset.csv
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"dataset.csv not found at {csv_path}")
except Exception as e:
    raise Exception(f"Error reading dataset.csv: {e}")

# 打印列名以调试
print("Columns in dataset.csv:")
print(df.columns.tolist())

# 查找 patient_id 和 filestem 列（忽略大小写和空格）
patient_col = None
filestem_col = None
for col in df.columns:
    col_clean = col.strip().lower()
    if col_clean == "patient_id":
        patient_col = col
    if col_clean == "filestem":
        filestem_col = col

# 检查是否找到必要的列
if not patient_col or not filestem_col:
    raise ValueError(f"Cannot find required columns. Expected 'patient_id' and 'filestem', but found: {df.columns.tolist()}")

# 按 patient_id 分组
patient_groups = df.groupby(patient_col).agg({filestem_col: list}).reset_index()

# 分割患者 ID：70% train, 20% valid, 10% test
train_patients, temp_patients = train_test_split(
    patient_groups, test_size=0.3, random_state=42
)
valid_patients, test_patients = train_test_split(
    temp_patients, test_size=0.3333, random_state=42
)

# 创建输出目录
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# 从多个图像目录中查找文件
def find_image_file(img_name):
    for image_dir in image_dirs:
        src_img = os.path.join(image_dir, img_name)
        if os.path.exists(src_img):
            return src_img
    return None

# 移动文件到对应目录
def move_files(patients, split, missing_files=[]):
    for _, row in patients.iterrows():
        for filestem in row[filestem_col]:
            # 图像文件（假设文件名为 filestem + ".png"）
            img_name = f"{filestem}.png"
            src_img = find_image_file(img_name)
            dst_img = os.path.join(output_dir, "images", split, img_name)
            if src_img:
                shutil.copy(src_img, dst_img)
            else:
                missing_files.append(img_name)
            # 标注文件
            label_name = f"{filestem}.txt"
            src_label = os.path.join(label_dir, label_name)
            dst_label = os.path.join(output_dir, "labels", split, label_name)
            if src_img:  # 只有找到图像时才处理标注
                if os.path.exists(src_label):
                    shutil.copy(src_label, dst_label)
                else:
                    with open(dst_label, "w") as f:
                        pass  # 创建空 .txt 文件
            else:
                missing_files.append(label_name)
    return missing_files

# 执行文件移动并收集缺失文件
missing_files = []
missing_files = move_files(train_patients, "train", missing_files)
missing_files = move_files(valid_patients, "valid", missing_files)
missing_files = move_files(test_patients, "test", missing_files)

# 打印缺失文件（如果有）
if missing_files:
    print(f"Warning: {len(missing_files)} files not found:")
    for f in missing_files[:5]:
        print(f"  {f}")
    if len(missing_files) > 5:
        print("  ... (more files omitted)")

# 保存分割后的患者列表
train_patients.to_csv(os.path.join(dataset_root, "train_patients.csv"), index=False)
valid_patients.to_csv(os.path.join(dataset_root, "valid_patients.csv"), index=False)
test_patients.to_csv(os.path.join(dataset_root, "test_patients.csv"), index=False)

# 打印统计信息
print(f"Train set: {len(train_patients)} patients, {sum(len(x) for x in train_patients[filestem_col])} images")
print(f"Valid set: {len(valid_patients)} patients, {sum(len(x) for x in valid_patients[filestem_col])} images")
print(f"Test set: {len(test_patients)} patients, {sum(len(x) for x in test_patients[filestem_col])} images")