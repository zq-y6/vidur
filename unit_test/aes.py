import torch
import extension_cpp

print("one block")
pt = torch.zeros(size=(4,), dtype=torch.uint32, device="cuda")
ct = torch.empty_like(pt)
iv = torch.zeros(size=(3,), dtype=torch.uint32, device="cuda")
rek = torch.zeros(size=(56,), dtype=torch.uint32, device="cuda")

ct = torch.ops.extension_cpp.my_paralell_aes_encrypt(pt, ct, iv, rek)
print(ct)


print("2048 blocks")
pt = torch.zeros(size=(2048 * 4,), dtype=torch.uint32, device="cuda")
ct = torch.empty_like(pt)

ct = torch.ops.extension_cpp.my_paralell_aes_encrypt(pt, ct, iv, rek)
print(ct)