from ultralytics import YOLO
import os
import json
import torch
import pandas as pd


def main():
    # 设置环境变量以减少内存碎片
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 设置自定义输出路径
    project_dir = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\custom_runs"# 使用绝对路径

    # 选择模型类型（可以在这里修改）
    model_type = "yolov8l"  # 选项

    # 根据模型类型设置模型名称
    if model_type == "yolov8n":
        model_name = "yolov8n"
        model_file = "yolov8n.pt"  # 直接使用文件名，Ultralytics 会自动下载
    elif model_type == "yolov8s":
        model_name = "yolov8s"
        model_file = "yolov8s.pt"  # 直接使用文件名，Ultralytics 会自动下载
    elif model_type == "yolov8m":
        model_name = "yolov8m"
        model_file = "yolov8m.pt"  # 直接使用文件名，Ultralytics 会自动下载
    elif model_type == "yolov8l":
        model_name = "yolov8l"
        model_file = "yolov8l.pt"  # 直接使用文件名，Ultralytics 会自动下载
    elif model_type == "yolov8x":
        model_name = "yolov8x"
        model_file = "yolov8x.pt"  # 直接使用文件名，Ultralytics 会自动下载
    elif model_type == "custom_best":
        model_name = "custom_model"
        model_file = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\custom_runs\yolov8n\weights\last.pt"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 设置实验名称为模型名称，确保不同模型保存到不同目录
    exp_name = model_name
    output_dir = os.path.join(project_dir, exp_name)

    # 加载预训练模型
    print(f"Loading model: {model_file}")
    model = YOLO(model_file)  # Ultralytics 会自动处理下载或加载

    # 打印设备信息
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 训练模型
    model.train(
        data=r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\meta.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        optimizer="SGD",
        lr0=0.001,
        cos_lr=True,
        patience=10,
        save=True,
        pretrained=True,
        project=project_dir,
        name=exp_name,
        exist_ok=True,
        save_period=2,
        verbose=True,
        amp=True,
        mixup=0.1,
        augment=True,
        iou=0.6,
        box=8.0,
        cls=1.5,
        degrees=15.0,
        shear=10.0,
        copy_paste=0.3
    )

    # 验证模型
    metrics = model.val()

    # 自定义保存验证结果
    results = {
        "model": model_name,
        "mAP@50": float(metrics.box.map50),
        "mAP@50:95": float(metrics.box.map),
        "precision": float(metrics.box.p),
        "recall": float(metrics.box.r)
    }

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存验证结果到 JSON 文件
    results_path = os.path.join(output_dir, "validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Validation results saved to {results_path}")

    # 保存到 result.csv
    result_csv_path = os.path.join(project_dir, "result.csv")
    results_df = pd.DataFrame([results])
    if os.path.exists(result_csv_path):
        results_df.to_csv(result_csv_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(result_csv_path, mode='w', header=True, index=False)
    print(f"Results appended to {result_csv_path}")

    # 打印验证结果
    print(f"mAP@50: {metrics.box.map50}")
    print(f"mAP@50:95: {metrics.box.map}")


if __name__ == "__main__":
    main()