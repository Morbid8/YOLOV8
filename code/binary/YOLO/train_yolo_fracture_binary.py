from ultralytics import YOLO
import os
import json
import torch
import pandas as pd


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    project_dir = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\custom_runs_binary"
    model_type = "yolov8s"

    if model_type == "yolov8n":
        model_name = "yolov8n"
        model_file = "yolov8n.pt"
    elif model_type == "yolov8s":
        model_name = "yolov8s"
        model_file = "yolov8s.pt"
    elif model_type == "yolov8m":
        model_name = "yolov8m"
        model_file = "yolov8m.pt"
    elif model_type == "custom_best":
        model_name = "custom_model"
        model_file = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\custom_runs_binary\my_exp\weights\best.pt"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    exp_name = model_name
    output_dir = os.path.join(project_dir, exp_name)

    print(f"Loading model: {model_file}")
    model = YOLO(model_file)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    yaml_path = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\meta_binary.yaml"

    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        optimizer="SGD",
        lr0=0.004,
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
        mixup=0.0,  # 禁用 mixup
        augment=False,  # 禁用所有增强
        iou=0.6,
        box=8.0,
        cls=3.0,
        degrees=0.0,  # 禁用旋转
        shear=0.0,  # 禁用剪切
        copy_paste=0.0  # 禁用 copy_paste
    )

    metrics = model.val(data=yaml_path)

    results = {
        "model": model_name,
        "mAP@50": float(metrics.box.map50),
        "mAP@50:95": float(metrics.box.map),
        "precision": float(metrics.box.p),
        "recall": float(metrics.box.r)
    }

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Validation results saved to {results_path}")

    result_csv_path = os.path.join(project_dir, "result.csv")
    results_df = pd.DataFrame([results])
    if os.path.exists(result_csv_path):
        results_df.to_csv(result_csv_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(result_csv_path, mode='w', header=True, index=False)
    print(f"Results appended to {result_csv_path}")

    print(f"mAP@50: {metrics.box.map50}")
    print(f"mAP@50:95: {metrics.box.map}")


if __name__ == "__main__":
    main()