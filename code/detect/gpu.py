import torch
print(torch.__version__)  # 查看 PyTorch 版本
# print(torch.version.cuda)  # 查看 PyTorch 对应的 CUDA 版本（如果有 GPU 支持）


print(torch.cuda.is_available())  # 应该返回 True
# print(torch.cuda.device_count())  # 应该返回 >=1
print(torch.cuda.get_device_name(0))  # 应该返回你的 GPU 型号（如 "NVIDIA GeForce RTX 3060"）

# 验证CUDA和PyTorch协同工作
x = torch.rand(5, 3).cuda()  # 将张量放入GPU
print(x.device)  # 应输出 "cuda:0"