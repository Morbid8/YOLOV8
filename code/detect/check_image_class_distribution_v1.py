import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 类别名称（与 meta.yaml 一致）
class_names = [
    "boneanomaly", "bonelesion", "foreignbody", "fracture",
    "metal", "periostealreaction", "pronatorsign", "softtissue", "text"
]

def check_image_class_distribution(label_dir, split_name, total_image_counter, total_images_per_split,
                                   times_distribution, split_image_counter, split_times_distribution):
    """
    统计每个异常类别出现在多少张图像中，计算图像占比，并统计出现次数分布。

    Args:
        label_dir (str): 标签文件目录（例如 'labels/train'）
        split_name (str): 子集名称（'train', 'valid', 'test'）
        total_image_counter (defaultdict): 记录所有子集的类别图像计数
        total_images_per_split (dict): 记录每个子集的总图像数
        times_distribution (defaultdict): 记录每个类别的出现次数分布（总体）
        split_image_counter (defaultdict): 记录当前子集的类别图像计数
        split_times_distribution (defaultdict): 记录当前子集的出现次数分布
    """
    image_count_per_class = defaultdict(int)  # 每个类别出现的图像数
    class_times_per_image = defaultdict(lambda: defaultdict(int))  # 每个类别的出现次数计数
    total_images = 0  # 当前子集的图像数

    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue

        total_images += 1
        label_path = os.path.join(label_dir, label_file)

        # 统计该图像中每个类别的出现次数
        class_counts_in_image = defaultdict(int)
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():  # 跳过空行
                        try:
                            class_id = int(line.strip().split()[0])
                            if class_id < len(class_names):
                                class_counts_in_image[class_id] += 1
                        except (IndexError, ValueError):
                            print(f"Warning: Invalid line in {label_file}: {line.strip()}")
                            continue
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
            continue

        # 记录图像计数和出现次数
        for class_id, count in class_counts_in_image.items():
            class_name = class_names[class_id]
            image_count_per_class[class_name] += 1
            total_image_counter[class_name] += 1
            split_image_counter[split_name][class_name] += 1
            # 按出现次数分类（0 次由总图像数计算，1 次、2 次、3 次及以上）
            times_key = 'More' if count >= 3 else str(count)
            class_times_per_image[class_name][times_key] += 1
            times_distribution[class_name][times_key] += 1
            split_times_distribution[split_name][class_name][times_key] += 1

    # 保存当前子集的总图像数
    total_images_per_split[split_name] = total_images

    # 打印当前子集的分布
    print(f"\nImage-based class distribution in {split_name} (Total images: {total_images}):")
    for class_id in range(len(class_names)):
        class_name = class_names[class_id]
        total_count = image_count_per_class.get(class_name, 0)
        percentage = (total_count / total_images) * 100 if total_images > 0 else 0

        print(f"  {class_name} (ID {class_id}):")
        print(f"    0 times: {total_images - total_count} images ({(total_images - total_count) / total_images * 100:.2f}%)")
        for times in ['1', '2', 'More']:
            count = class_times_per_image[class_name].get(times, 0)
            percentage_times = (count / total_images) * 100 if total_images > 0 else 0
            times_label = f"{times} time{'s' if times == '2' else ''}" if times != 'More' else 'More (3+ times)'
            print(f"    {times_label}: {count} images ({percentage_times:.2f}%)")
        print(f"    Total: {total_count} images ({percentage:.2f}%)")

def plot_class_distribution(image_counter, total_images, times_distribution, output_dir, split_name=None):
    """
    绘制类别分布图表。

    Args:
        image_counter (defaultdict): 类别图像计数（总体或某个子集）
        total_images (int): 总图像数（总体或某个子集）
        times_distribution (defaultdict): 出现次数分布（总体或某个子集）
        output_dir (str): 图表保存目录
        split_name (str, optional): 子集名称（'train', 'valid', 'test' 或 None 表示总体）
    """
    prefix = f"{split_name}_" if split_name else "all_"
    title_suffix = f" ({split_name})" if split_name else " (All Splits)"

    # 图表 1：每个类别的总图像占比（柱状图）
    percentages = [(image_counter[class_name] / total_images * 100) for class_name in class_names]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, percentages, color='skyblue')
    plt.title(f'Total Image-based Class Distribution{title_suffix}')
    plt.xlabel('Class')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}class_distribution_total.png'))
    plt.close()

    # 图表 2：每个类别的出现次数分布（堆叠柱状图）
    times_labels = ['0 times', '1 time', '2 times', 'More (3+ times)']
    data = {times_label: [] for times_label in times_labels}

    for class_name in class_names:
        total_count = image_counter.get(class_name, 0)
        # 0 次的图像数
        data['0 times'].append((total_images - total_count) / total_images * 100)
        # 1 次、2 次、More 的图像数
        for times in ['1', '2', 'More']:
            count = times_distribution[class_name].get(times, 0)
            percentage = (count / total_images * 100) if total_images > 0 else 0
            times_label = f"{times} time{'s' if times == '2' else ''}" if times != 'More' else 'More (3+ times)'
            data[times_label].append(percentage)

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(class_names))
    for times_label in times_labels:
        plt.bar(class_names, data[times_label], bottom=bottom, label=times_label)
        bottom += np.array(data[times_label])

    plt.title(f'Class Distribution by Occurrence Frequency{title_suffix}')
    plt.xlabel('Class')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}class_distribution_by_frequency.png'))
    plt.close()

def save_distribution_to_csv(total_image_counter, total_images_all, times_distribution,
                            split_image_counter, total_images_per_split, split_times_distribution, output_dir):
    """
    将类别分布和出现次数分布保存为 CSV 文件。

    Args:
        total_image_counter (defaultdict): 总体类别图像计数
        total_images_all (int): 总体图像数
        times_distribution (defaultdict): 总体出现次数分布
        split_image_counter (defaultdict): 各子集类别图像计数
        total_images_per_split (dict): 各子集总图像数
        split_times_distribution (defaultdict): 各子集出现次数分布
        output_dir (str): 保存目录
    """
    splits = ['all', 'train', 'valid', 'test']
    times_labels = ['0 times', '1 time', '2 times', 'More (3+ times)']
    data = []

    for split in splits:
        if split == 'all':
            image_counter = total_image_counter
            total_images = total_images_all
            times_dist = times_distribution
        else:
            image_counter = split_image_counter[split]
            total_images = total_images_per_split[split]
            times_dist = split_times_distribution[split]

        for class_name in class_names:
            total_count = image_counter.get(class_name, 0)
            row = {'Split': split, 'Class': class_name}
            # 总占比
            row['Total Images'] = total_count
            row['Total Percentage (%)'] = (total_count / total_images * 100) if total_images > 0 else 0
            # 出现次数分布
            for times in ['0 times'] + times_labels[1:]:
                if times == '0 times':
                    count = total_images - total_count
                else:
                    times_key = times.split()[0] if times != 'More (3+ times)' else 'More'
                    count = times_dist[class_name].get(times_key, 0)
                row[f'{times} Images'] = count
                row[f'{times} Percentage (%)'] = (count / total_images * 100) if total_images > 0 else 0
            data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'class_distribution.csv'), index=False)
    print(f"\nClass distribution saved to {os.path.join(output_dir, 'class_distribution.csv')}")

if __name__ == "__main__":
    dataset_root = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX"
    total_image_counter = defaultdict(int)  # 所有子集的类别图像计数
    total_images_per_split = {}  # 每个子集的总图像数
    times_distribution = defaultdict(lambda: defaultdict(int))  # 每个类别的出现次数分布（总体）
    split_image_counter = defaultdict(lambda: defaultdict(int))  # 每个子集的类别图像计数
    split_times_distribution = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # 每个子集的出现次数分布

    # 统计每个子集的分布
    for split in ['train', 'valid', 'test']:
        check_image_class_distribution(
            os.path.join(dataset_root, "data", "labels", split), split,
            total_image_counter, total_images_per_split, times_distribution,
            split_image_counter, split_times_distribution
        )

    # 汇总统计
    total_images_all = sum(total_images_per_split.values())
    print(f"\n📊 Total image-based class distribution across all splits (Total images: {total_images_all}):")
    for class_id in range(len(class_names)):
        class_name = class_names[class_id]
        total_count = total_image_counter.get(class_name, 0)
        percentage = (total_count / total_images_all) * 100 if total_images_all > 0 else 0

        print(f"  {class_name} (ID {class_id}):")
        print(f"    0 times: {total_images_all - total_count} images ({(total_images_all - total_count) / total_images_all * 100:.2f}%)")
        for times in ['1', '2', 'More']:
            count = times_distribution[class_name].get(times, 0)
            percentage_times = (count / total_images_all) * 100 if total_images_all > 0 else 0
            times_label = f"{times} time{'s' if times == '2' else ''}" if times != 'More' else 'More (3+ times)'
            print(f"    {times_label}: {count} images ({percentage_times:.2f}%)")
        print(f"    Total: {total_count} images ({percentage:.2f}%)")

    # 绘制图表（总体和每个子集）
    output_dir = os.path.join(dataset_root, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # 总体图表
    plot_class_distribution(total_image_counter, total_images_all, times_distribution, output_dir)
    # 每个子集的图表
    for split in ['train', 'valid', 'test']:
        plot_class_distribution(split_image_counter[split], total_images_per_split[split],
                               split_times_distribution[split], output_dir, split)

    # 保存分布数据到 CSV
    save_distribution_to_csv(total_image_counter, total_images_all, times_distribution,
                            split_image_counter, total_images_per_split, split_times_distribution, output_dir)
    print(f"\nPlots saved to {output_dir}")