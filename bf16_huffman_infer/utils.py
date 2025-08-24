import copy
from typing import Optional
import torch
from torch import Tensor, nn


def split_bf16(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dtype == torch.bfloat16
    x = x.view(torch.int16)
    exp = (x >> 7)[..., None].view(torch.uint8)[..., 0].contiguous()
    rem = (((x >> 8) & 0x80) + (x & 0x7f))[..., None].view(torch.uint8)[..., 0].contiguous()
    return rem, exp


def get_block_size_n(x: torch.Tensor) -> Optional[tuple[int, int]]:
    n = x.size(1)
    
    if n % 128 == 0:
        n //= 128
        for i in range(4096 // 128, 2, -1):
            if n % i == 0 and n // i <= 8:
                return (n // i, )
        for i in range(4096 // 128, 16384 // 128 + 1):
            if n % i == 0 and n // i <= 32:
                return (n // i, )
        for i in range(4096 // 128, 2, -1):
            if n % i == 0 and n // i <= 32:
                return (n // i, )
    
    # return None
    raise RuntimeError(f"Could not find suitable block size for shape {x.shape}")


def get_model_size(model: nn.Module) -> int:
    total_size = 0
    for name, param in model.named_parameters():
        size = param.nelement() * param.element_size()
        total_size += size
    for name, param in model.named_buffers():
        size = param.nelement() * param.element_size()
        total_size += size
    return total_size


def print_stats(ori: Tensor, new: nn.Module):
    ori_size = ori.nelement() * ori.element_size()
    new_size = sum(buffer.nelement() * buffer.element_size() for buffer in new.buffers())
    
    print(f'ori size = {ori_size / 1024 ** 2:.2f}MiB')
    print(f'new size = {new_size / 1024 ** 2:.2f}MiB')
    print(f'compression ratio = {new_size / ori_size:.2%}')
    print(f'bits = {new_size / ori_size * 16:.2f}')


def shallow_copy_model(model: nn.Module) -> nn.Module:
    memo = {}
    for p in model.parameters():
        memo[id(p)] = p
    for b in model.buffers():
        memo[id(b)] = b
    return copy.deepcopy(model, memo=memo)

