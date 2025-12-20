import sys
print("=== PyTorch测试开始 ===")
print(f"Python版本: {sys.version.split()[0]}")
print(f"Python路径: {sys.executable}")

print("\n1. 尝试导入torch...")
try:
    import torch
    print("   ✓ torch导入成功")
    print(f"   版本: {torch.__version__}")
except ImportError as e:
    print(f"   ✗ 导入失败: {e}")
    sys.exit(1)

print("\n2. 检查CUDA...")
try:
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA可用: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"   GPU数量: {device_count}")
        for i in range(device_count):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("   CUDA不可用，将在CPU模式下运行")
except Exception as e:
    print(f"   ✗ CUDA检查失败: {e}")

print("\n3. 测试基本张量运算...")
try:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"   使用设备: {device}")
    else:
        device = torch.device('cpu')
        print(f"   使用设备: {device}")
    
    x = torch.rand(2, 3).to(device)
    y = torch.rand(3, 2).to(device)
    z = torch.mm(x, y)
    
    print(f"   矩阵乘法成功!")
    print(f"   输入形状: {x.shape} @ {y.shape}")
    print(f"   输出形状: {z.shape}")
    print(f"   结果示例:\n{z}")
except Exception as e:
    print(f"   ✗ 张量运算失败: {e}")

print("\n=== PyTorch测试结束 ===")
