import pandas as pd
import os

dataset_root = r"D:\pythonProject\ultralytics-main\GRAZPEDWRI-DX"
binary_dir = os.path.join(dataset_root, "binary_classification")

subsets = ["train", "valid", "test"]
for subset in subsets:
    csv_path = os.path.join(binary_dir, f"{subset}_labels.csv")
    df = pd.read_csv(csv_path)
    total = len(df)
    positive = len(df[df["label"] == 1])
    negative = len(df[df["label"] == 0])
    print(f"{subset} set:")
    print(f"  Total images: {total}")
    print(f"  Positive (fracture): {positive} ({positive/total:.2%})")
    print(f"  Negative (no fracture): {negative} ({negative/total:.2%})")