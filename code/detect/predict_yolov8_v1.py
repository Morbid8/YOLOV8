from ultralytics import YOLO
import os
import cv2
import numpy as np
from datetime import datetime


def is_grayscale(image):
    """检查图像是否为灰度图"""
    return len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)


def convert_to_grayscale(image_path, output_path):
    """将图像转换为灰度图并保存"""
    print(f"Processing: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return False

    print(f"Original image shape: {img.shape}")

    # 如果已经是灰度图，直接保存
    if is_grayscale(img):
        print(f"Image {image_path} is already grayscale.")
        gray_img = img
    else:
        print(f"Converting {image_path} to grayscale...")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 确保是3通道灰度图（兼容YOLO输入）
    if len(gray_img.shape) == 2:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    success = cv2.imwrite(output_path, gray_img)
    print(f"Save {'successful' if success else 'failed'}: {output_path}")
    return success


def main():
    # 加载训练好的模型
    model_path = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\custom_runs\yolov8s\weights\best.pt"
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    if model is None:
        print("Failed to load model!")
        exit()

    # 设置路径
    source_dir = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\assets"
    output_dir = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\assets_grayscale"
    print(f"Source directory: {source_dir}")
    print(f"Grayscale output directory: {output_dir}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 转换所有图像为灰度图
    grayscale_images = []
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            input_path = os.path.join(source_dir, filename)
            output_path = os.path.join(output_dir, filename)
            if convert_to_grayscale(input_path, output_path):
                grayscale_images.append(output_path)

    if not grayscale_images:
        print("No images available for prediction.")
        exit()

    # 生成时间戳和输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"predict_{timestamp}"
    output_project = "runs/detect"  # YOLO默认输出目录

    print(f"\nStarting prediction on {len(grayscale_images)} images...")
    print(f"Results will be saved to: {os.path.join(output_project, output_name)}")

    # 执行预测（从grayscale目录读取）
    results = model.predict(
        source=output_dir,
        save=True,
        imgsz=416,
        conf=0.25,
        iou=0.45,
        device=0,
        project=output_project,
        name=output_name,
        exist_ok=True
    )

    # 打印预测结果
    print("\nPrediction summary:")
    for i, result in enumerate(results):
        print(f"\nImage {i + 1}: {os.path.basename(result.path)}")
        if result.boxes is not None:
            print(f"Detected {len(result.boxes)} objects")
            classes = result.boxes.cls.cpu().numpy()
            unique_classes, counts = np.unique(classes, return_counts=True)
            class_distribution = dict(zip(unique_classes, counts))
            print("Class distribution:", class_distribution)

    print(f"\nAll results saved to: {os.path.join(output_project, output_name)}")
    print("Prediction completed successfully!")


if __name__ == "__main__":
    main()