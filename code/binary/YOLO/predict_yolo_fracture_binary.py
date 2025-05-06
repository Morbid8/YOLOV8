from ultralytics import YOLO
import os
import json
import torch
import cv2
from datetime import datetime


def main():
    # 设置路径
    project_dir = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\code\binary\YOLO"
    model_name = "yolov8s"
    best_model_path = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\custom_runs_binary\yolov8s\weights\best.pt"

    # 输入图像目录
    custom_image_dir = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\assets_grayscale"

    # 使用YOLO默认输出目录结构
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"predict_{timestamp}"
    output_project = "runs/binary"  # YOLO默认输出目录

    # 输出路径
    test_results_path = os.path.join(output_project, output_name, "test_results.json")
    visualization_dir = os.path.join(output_project, output_name)

    # 加载模型
    print(f"Loading model: {best_model_path}")
    model = YOLO(best_model_path)

    # 打印 GPU 信息
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 推理参数
    device = 0 if torch.cuda.is_available() else "cpu"
    imgsz = 640
    conf = 0.3
    iou = 0.6

    # 收集图像
    test_images = [os.path.join(custom_image_dir, f) for f in os.listdir(custom_image_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not test_images:
        print(f"No images found in {custom_image_dir}")
        return

    print(f"Found {len(test_images)} images in {custom_image_dir}. Starting inference...")

    # 推理并保存预测结果
    results_list = []

    # 使用model.predict自动保存到指定目录
    results = model.predict(
        source=custom_image_dir,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        project=output_project,
        name=output_name,
        save=True,  # 自动保存可视化结果
        exist_ok=True
    )

    # 处理预测结果
    for result in results:
        img_name = os.path.basename(result.path)
        predictions = []
        has_fracture = False

        if result.boxes is not None:
            boxes = result.boxes.xywhn.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes):
                predictions.append({
                    "class_id": int(cls),
                    "class_name": "fracture",
                    "confidence": float(score),
                    "box": [float(x) for x in box]
                })

            if len(predictions) > 0:
                has_fracture = True

        results_list.append({
            "image": img_name,
            "has_fracture": has_fracture,
            "predictions": predictions
        })

    # 保存预测结果为JSON
    os.makedirs(os.path.join(output_project, output_name), exist_ok=True)
    with open(test_results_path, "w") as f:
        json.dump(results_list, f, indent=4)

    print(f"Test results saved to {os.path.abspath(test_results_path)}")
    print(f"Visualized images saved to {os.path.abspath(visualization_dir)}")


if __name__ == "__main__":
    main()