import os
from collections import Counter

# 类别名称（与 meta.yaml 一致）
class_names = [
    "boneanomaly", "bonelesion", "foreignbody", "fracture",
    "metal", "periostealreaction", "pronatorsign", "softtissue", "text"
]

# 统计每个子集的类别分布（统计的是一种异常占所有异常的占比，但是忽略了一张图片可能有好几个异常）
def check_class_distribution(label_dir, split_name, total_counter):
    class_counts = Counter()
    for label_file in os.listdir(label_dir):
        with open(os.path.join(label_dir, label_file), "r") as f:
            for line in f:
                if line.strip():  # 跳过空行
                    class_id = int(line.strip().split()[0])
                    class_counts[class_id] += 1
                    total_counter[class_id] += 1  # 累加到总计中

    print(f"\nClass distribution in {split_name}:")
    for class_id in range(len(class_names)):
        count = class_counts.get(class_id, 0)
        print(f"  {class_names[class_id]} (ID {class_id}): {count} instances")

# 主程序
if __name__ == "__main__":
    dataset_root = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX"
    total_counter = Counter()

    check_class_distribution(os.path.join(dataset_root, "data", "labels", "train"), "train", total_counter)
    check_class_distribution(os.path.join(dataset_root, "data", "labels", "valid"), "valid", total_counter)
    check_class_distribution(os.path.join(dataset_root, "data", "labels", "test"), "test", total_counter)

    # 汇总统计
    print("\n📊 Total class distribution across all splits:")
    total_instances = sum(total_counter.values())
    for class_id in range(len(class_names)):
        count = total_counter.get(class_id, 0)
        percent = (count / total_instances) * 100 if total_instances else 0
        print(f"  {class_names[class_id]} (ID {class_id}): {count} instances ({percent:.2f}%)")

    print(f"\nTotal instances: {total_instances}")
