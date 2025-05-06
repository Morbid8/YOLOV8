import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch.cuda.amp import GradScaler, autocast
import pynvml


# 自定义数据集类
class FractureDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        print(f"Loading CSV file: {csv_file}")
        self.data = pd.read_csv(csv_file)
        print(f"Loaded {len(self.data)} samples from {csv_file}")
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["image_path"]
        label = self.data.iloc[idx]["label"]

        img = Image.open(img_path).convert("L")
        img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long), img_path


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])


# GPU 利用率监控
def get_gpu_usage():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return info.gpu
    except Exception as e:
        return f"Error: {e}"


# 主函数
if __name__ == '__main__':
    print("Starting script...")

    # 加载数据集
    dataset_root = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX"
    binary_dir = os.path.join(dataset_root, "binary_classification")
    print(f"Dataset root: {dataset_root}")
    print(f"Binary dir: {binary_dir}")

    train_dataset = FractureDataset(os.path.join(binary_dir, "train_labels.csv"), transform=transform)
    valid_dataset = FractureDataset(os.path.join(binary_dir, "valid_labels.csv"), transform=transform)
    test_dataset = FractureDataset(os.path.join(binary_dir, "test_labels.csv"), transform=transform)

    print("Creating DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    print("DataLoader created.")

    # 模型定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_name = "resnet18"
    print("Loading model...")
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    print("Model loaded.")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    # 训练和验证
    num_epochs = 100
    best_acc = 0.0
    output_dir = os.path.join(dataset_root, "binary_classification", "results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}], GPU Usage: {get_gpu_usage()}%")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}]")

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, _ in valid_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(valid_loader)
        val_acc = 100 * correct / total

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Saved best model with Val Acc: {best_acc:.2f}%")

    print("Starting testing...")
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
    model.eval()
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for images, labels, img_paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for img_path, true_label, pred_label in zip(img_paths, labels.cpu().numpy(), predicted.cpu().numpy()):
                predictions.append({
                    "image_path": img_path,
                    "true_label": true_label,
                    "predicted_label": pred_label
                })

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    print(f"Predictions saved to {os.path.join(output_dir, 'predictions.csv')}")

    results = {
        "model": model_name,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc
    }
    pd.DataFrame([results]).to_csv(os.path.join(output_dir, "results.csv"), index=False)
    print(f"Results saved to {os.path.join(output_dir, 'results.csv')}")