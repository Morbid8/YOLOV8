import os
import pandas as pd

# 数据集路径
dataset_root = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX"
image_base_dir = os.path.join(dataset_root, "data", "images")
label_base_dir = os.path.join(dataset_root, "data", "labels")
output_dir = os.path.join(dataset_root, "binary_classification")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# fracture 类别的 ID
FRACTURE_CLASS_ID = 3

# 处理每个子集（train、val、test）
def generate_labels_for_subset(subset):
    image_dir = os.path.join(image_base_dir, subset)
    label_dir = os.path.join(label_base_dir, subset)
    data = []

    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        # 对应的图像文件
        img_name = label_file.replace('.txt', '.png')
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        # 读取标签文件
        label_path = os.path.join(label_dir, label_file)
        has_fracture = False
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.strip():
                    continue
                class_id = int(line.strip().split()[0])
                if class_id == FRACTURE_CLASS_ID:
                    has_fracture = True
                    break
        
        # 标签：1（有骨折），0（无骨折）
        label = 1 if has_fracture else 0
        data.append({"image_path": img_path, "label": label})

    # 保存到 CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, f"{subset}_labels.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {subset} labels to {csv_path}")
    return df

# 生成所有子集的标签
subsets = ["train", "valid", "test"]
for subset in subsets:
    generate_labels_for_subset(subset)