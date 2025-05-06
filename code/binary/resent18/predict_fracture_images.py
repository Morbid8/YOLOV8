import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch.amp import autocast
from PIL import Image
import pandas as pd
from pathlib import Path
from datetime import datetime

# 数据预处理（与训练时一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])

# 主函数
if __name__ == '__main__':
    print("Starting prediction script...")

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    print("Loading model...")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model_path = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\binary_classification\results\best_model.pth"
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # 测试图片文件夹
    test_images_dir = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\assets"
    output_dir = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX\binary_classification\results"  # 修正为Windows路径
    os.makedirs(output_dir, exist_ok=True)
    print(f"Test images directory: {test_images_dir}")
    print(f"Output directory: {output_dir}")

    # 收集所有图片路径
    image_extensions = (".png", ".jpg", ".jpeg")
    image_paths = [str(p) for p in Path(test_images_dir).rglob("*") if p.suffix.lower() in image_extensions]
    print(f"Found {len(image_paths)} images to predict.")

    # 预测结果存储
    predictions = []

    # 预测每张图片
    for img_path in image_paths:
        try:
            # 加载并预处理图片
            img = Image.open(img_path).convert("L").convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                with autocast('cuda'):
                    output = model(img)
                    _, predicted = torch.max(output, 1)
                    predicted_label = predicted.item()

            # 转换为人类可读的标签
            label_str = "Fracture" if predicted_label == 1 else "No Fracture"
            print(f"Image: {os.path.basename(img_path)}, Predicted: {label_str}")

            predictions.append({
                "image_path": img_path,
                "predicted_label": predicted_label,
                "label_str": label_str
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # 生成带时间戳的唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f"predictions_{timestamp}.csv")

    # 保存预测结果
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    # 验证文件是否创建
    if os.path.exists(output_csv):
        print(f"File verification: {output_csv} exists, size: {os.path.getsize(output_csv)} bytes")
    else:
        print("Error: Failed to create output file!")