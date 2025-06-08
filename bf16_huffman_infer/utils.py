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
    
    for b in range(12, 7, -1):
        power_of_2 = 2 ** b
        
        if n % power_of_2 == 0:
            split_n = n // power_of_2
            return (split_n, power_of_2)
    
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

