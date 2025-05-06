import os
from collections import defaultdict

# 路径设置
dataset_root = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX"
labels_root = os.path.join(dataset_root, "data", "labels")
splits = ["train", "valid", "test"]

# 类名映射
class_names = {
    0: "boneanomaly",
    1: "bonelesion",
    2: "foreignbody",
    3: "fracture",
    4: "metal",
    5: "periostealreaction",
    6: "pronatorsign",
    7: "softtissue",
    8: "text"
}

# 统计结果
total_stats = {"total_images": 0, "class_images": defaultdict(int)}

# 统计每个子集中每类的图片数量
for split in splits:
    print(f"Analyzing {split} split...")

    labels_dir = os.path.join(labels_root, split)
    if not os.path.exists(labels_dir):
        print(f"Error: {labels_dir} does not exist")
        continue

    total_images = 0
    class_images = defaultdict(int)

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        total_images += 1
        label_file_path = os.path.join(labels_dir, label_file)

        # 读取 labels 文件
        with open(label_file_path, "r") as f:
            lines = f.readlines()

        # 记录该图片中出现的类别
        classes_in_image = set()
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                classes_in_image.add(class_id)

        # 统计每类出现的图片数量
        for class_id in classes_in_image:
            class_images[class_id] += 1

    # 计算占比
    print(f"Total images: {total_images}")
    for class_id in sorted(class_images.keys()):
        ratio = class_images[class_id] / total_images * 100
        print(f"{class_names[class_id]} (class_id={class_id}): {class_images[class_id]} images, ratio: {ratio:.2f}%")
    print()

    # 累加到总体统计
    total_stats["total_images"] += total_images
    for class_id, count in class_images.items():
        total_stats["class_images"][class_id] += count

# 总体统计
print("Overall statistics:")
print(f"Total images: {total_stats['total_images']}")
for class_id in sorted(total_stats["class_images"].keys()):
    ratio = total_stats["class_images"][class_id] / total_stats["total_images"] * 100
    print(
        f"{class_names[class_id]} (class_id={class_id}): {total_stats['class_images'][class_id]} images, ratio: {ratio:.2f}%")