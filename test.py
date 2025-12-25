import torch
import time

def main():
    # 1) 检查 CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请确认已安装 GPU 驱动和支持 CUDA 的 PyTorch")

    # 2) 强制使用 cuda:2
    device = torch.device("cuda:2")

    # 3) 打印设备信息
    print("Using device:", device)
    print("GPU name:", torch.cuda.get_device_name(2))

    # 4) 一个小计算：矩阵乘法
    a = torch.rand(1024, 1024, device=device)
    b = torch.rand(1024, 1024, device=device)

    # 让 GPU 先热身一下
    torch.cuda.synchronize(device)
    t0 = time.time()

    c = a @ b  # 矩阵乘法

    # 同步一下确保计算完成再计时
    torch.cuda.synchronize(device)
    t1 = time.time()

    print("Result tensor device:", c.device)
    print("c[0,0] =", c[0, 0].item())
    print("Time (ms):", (t1 - t0) * 1000)

if __name__ == "__main__":
    main()
