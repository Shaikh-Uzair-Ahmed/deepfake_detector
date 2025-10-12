import torch 

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")

    # 🔥 One-liner GPU compute test
    x = torch.rand(10000, 10000, device="cuda")
    print("✅ GPU computation successful:", torch.sum(x).item())
else:
    print("❌ CUDA not available — running on CPU.")