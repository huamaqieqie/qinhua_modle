import torch

print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
print(dd)
print(zjm，test,test)
# 测试张量计算
if torch.cuda.is_available():
    device = torch.device('cuda:0')  # 使用第一块GPU
    x = torch.rand(5, 3).to(device)
    y = torch.rand(3, 4).to(device)
    z = torch.mm(x, y)
    print(f'GPU计算成功，结果形状: {z.shape}')
    print(f'使用的GPU: {torch.cuda.get_device_name(0)}')