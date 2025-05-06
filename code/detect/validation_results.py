from ultralytics import YOLO
import os
import csv
import json
from collections import defaultdict

def get_ground_truth_classes(label_dir, class_names):
    """统计验证集每类别的图像数和实例数"""
    images_per_class = defaultdict(int)
    instances_per_class = defaultdict(int)
    image_has_class = defaultdict(set)

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        image_name = label_file
        with open(os.path.join(label_dir, label_file), "r") as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    instances_per_class[class_id] += 1
                    image_has_class[class_id].add(image_name)

    ground_truth = {
        "images": {name: 0 for name in class_names.values()},
        "instances": {name: 0 for name in class_names.values()}
    }
    for class_id in class_names:
        name = class_names[class_id]
        ground_truth["images"][name] = len(image_has_class[class_id])
        ground_truth["instances"][name] = instances_per_class[class_id]

    return ground_truth

def validate_model():
    # 配置路径
    model_path = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\custom_runs\yolov8l\weights\best.pt"
    model_name = "yolov8s"
    output_dir = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\custom_runs\yolov8l"
    label_dir = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\data\labels\valid"

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    model = YOLO(model_path)
    print(f"Loaded model from {model_path}")

    # 获取验证集真值类别分布
    class_names = model.names
    ground_truth = get_ground_truth_classes(label_dir, class_names)
    print("Ground truth class distribution (images):", ground_truth["images"])
    print("Ground truth class distribution (instances):", ground_truth["instances"])

    results = model.val(
        data=r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\meta.yaml",
        workers=0,
        conf=0.01,
        save=False,
        save_json=False,
        save_hybrid=False,
        plots=False
    )

    # 获取指标
    metrics = results.box
    total_classes = len(class_names)

    # 打印 metrics.ap_class_index 以调试
    print("Metrics class indices:", metrics.ap_class_index)

    # 保存为 CSV 格式
    csv_data = [["Class", "Images", "Instances", "Precision", "Recall", "mAP50", "mAP50-95"]]

    # 添加总览（all）
    total_instances = sum(ground_truth["instances"].values())
    total_images = len([f for f in os.listdir(label_dir) if f.endswith(".txt")])
    csv_data.append([
        "all",
        total_images,
        total_instances,
        round(metrics.mp, 3),
        round(metrics.mr, 3),
        round(metrics.map50, 3),
        round(metrics.map, 3)
    ])

    # 准备 JSON 结构
    json_data = {
        "summary": {
            "Precision": float(metrics.mp),
            "Recall": float(metrics.mr),
            "mAP50": float(metrics.map50),
            "mAP50-95": float(metrics.map)
        },
        "per_class": []
    }

    # 遍历所有类
    for class_id in range(total_classes):
        name = class_names[class_id]
        images = ground_truth["images"][name]
        instances = ground_truth["instances"][name]

        # 如果类别无真值，指标置为 0
        if instances == 0:
            precision = recall = ap50 = ap = 0.0
        else:
            # 检查 class_id 是否在 metrics.ap_class_index 中
            if class_id in metrics.ap_class_index:
                idx = list(metrics.ap_class_index).index(class_id)
                precision = float(metrics.p[idx]) if idx < len(metrics.p) else 0.0
                recall = float(metrics.r[idx]) if idx < len(metrics.r) else 0.0
                ap50 = float(metrics.ap50[idx]) if idx < len(metrics.ap50) else 0.0
                ap = float(metrics.ap[idx]) if idx < len(metrics.ap) else 0.0
            else:
                precision = recall = ap50 = ap = 0.0

        csv_data.append([
            name,
            images,
            instances,
            round(precision, 3),
            round(recall, 3),
            round(ap50, 3),
            round(ap, 3)
        ])
        json_data["per_class"].append({
            "class": name,
            "images": images,
            "instances": instances,
            "precision": precision,
            "recall": recall,
            "mAP50": ap50,
            "mAP50-95": ap
        })

    # 保存 CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "validation_results.csv")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    print(f"CSV results saved to {csv_path}")

    # 保存 JSON
    json_path = os.path.join(output_dir, "validation_results.json")
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)
    print(f"JSON results saved to {json_path}")

    # 打印 JSON 数据
    print(json.dumps(json_data, indent=4))

if __name__ == '__main__':
    validate_model()