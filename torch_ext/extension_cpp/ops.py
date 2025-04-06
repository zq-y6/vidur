import torch
from torch import Tensor

__all__ = ["my_paralell_aes_encrypt"]


def my_paralell_aes_encrypt(pt: Tensor, ct: Tensor, iv: Tensor, rek: Tensor) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.extension_cpp.my_paralell_aes_encrypt.default(pt, ct, iv, rek)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("extension_cpp::my_paralell_aes_encrypt")
def _(pt, ct, iv, rek):
    torch._check(pt.device == rek.device)
    torch._check(rek.device == iv.device)
    torch._check(iv.device == ct.device)
    return torch.empty_like(pt)
