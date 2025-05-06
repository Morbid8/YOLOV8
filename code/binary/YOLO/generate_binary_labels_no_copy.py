import os
import pandas as pd

# 路径设置
dataset_root = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX"
binary_dir = os.path.join(dataset_root, "binary_classification")
gray_images_dir = os.path.join(dataset_root, "gray_images")  # 灰度处理后的图片根目录
labels_root = os.path.join(dataset_root, "data", "labels")  # 原始 9 分类 labels
binary_labels_root = os.path.join(dataset_root, "data_binary_yolo", "labels")  # 二分类 labels（带 yolo 标识）
splits = ["train", "valid", "test"]

# 创建二分类 labels 目录
for split in splits:
    os.makedirs(os.path.join(binary_labels_root, split), exist_ok=True)

# 遍历 gray_images 子文件夹，构建图片路径映射
image_path_mapping = {}
subdirs = [f"images_part{i}" for i in range(1, 5)]  # images_part1 到 images_part4
for subdir in subdirs:
    subdir_path = os.path.join(gray_images_dir, subdir)
    if not os.path.exists(subdir_path):
        continue
    for img_name in os.listdir(subdir_path):
        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(subdir_path, img_name)
            img_name_no_ext = os.path.splitext(img_name)[0]
            image_path_mapping[img_name_no_ext] = img_path


# 转换 9 分类标签为二分类（仅保留 fracture）
def convert_label_file(src_label_file, dst_label_file):
    if not os.path.exists(src_label_file) or os.path.getsize(src_label_file) == 0:
        # 如果文件不存在或为空，表示无目标，保持为空
        with open(dst_label_file, "w") as f:
            pass
        return

    # 读取原始标签
    with open(src_label_file, "r") as f:
        lines = f.readlines()

    # 仅保留 class_id == 3 的行，并将 class_id 改为 0
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        if class_id == 3:  # fracture
            new_line = f"0 {' '.join(parts[1:])}\n"
            new_lines.append(new_line)

    # 写入新标签文件
    with open(dst_label_file, "w") as f:
        f.writelines(new_lines)


# 为每个分割转换 labels 文件
for split in splits:
    print(f"Processing {split} split...")

    # 读取 labels.csv
    csv_file = os.path.join(binary_dir, f"{split}_labels.csv")
    data = pd.read_csv(csv_file)


    # 更新 labels.csv 中的 image_path，指向 gray_images 子文件夹
    def update_image_path(old_path):
        img_name = os.path.splitext(os.path.basename(old_path))[0]
        return image_path_mapping.get(img_name, old_path)  # 如果找不到，保持原路径


    data["image_path"] = data["image_path"].apply(update_image_path)

    # 保存更新后的 labels.csv
    data.to_csv(csv_file, index=False)
    print(f"Updated {csv_file} with new image paths")

    # 转换 labels 文件
    src_labels_dir = os.path.join(labels_root, split)
    dst_labels_dir = os.path.join(binary_labels_root, split)
    for idx, row in data.iterrows():
        img_path = row["image_path"]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        src_label_file = os.path.join(src_labels_dir, f"{img_name}.txt")
        dst_label_file = os.path.join(dst_labels_dir, f"{img_name}.txt")
        convert_label_file(src_label_file, dst_label_file)

    print(f"Generated binary labels for {split} split, saved to {dst_labels_dir}")