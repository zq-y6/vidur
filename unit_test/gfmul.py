import torch
import extension_cpp

print("one block")
msg = torch.zeros(size=(2,), dtype=torch.uint64, device="cuda")
msg[0] = 0b1011
res = torch.empty_like(msg)
h   = torch.zeros(size=(2,), dtype=torch.uint64, device="cuda")
h[0] = 0b1101

res = torch.ops.extension_cpp.my_paralell_gf128mul(msg, h, res, 1)
print(res)


print("2048 blocks")
msg = torch.zeros(size=(2 * 2048,), dtype=torch.uint64, device="cuda")
msg[0] = 0b1011
msg[1] = 0b1011
res = torch.empty_like(msg)
h   = torch.zeros(size=(2,), dtype=torch.uint64, device="cuda")
h[0] = 0b1101

res = torch.ops.extension_cpp.my_paralell_gf128mul(msg, h, res, 1024)
print(res)