import torch, time
a = torch.randn(8192, 8192, device="cuda")
b = torch.randn(8192, 8192, device="cuda")
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    torch.matmul(a, b)
torch.cuda.synchronize()
print(time.time() - start)