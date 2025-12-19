from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from tqdm import tqdm

from .huffman import LUTHuffmanEncoder
from .ans import RANSEncoder

from .utils import get_block_size_n, split_bf16

from .ops import ans_decode, ans_encode, gemv_bf16_ans, gemv_bf16_huffman, huffman_decode, huffman_encode

OP_PER_LANE = 1

__all__ = [
    "linear_huffman",
    "HuffmanWeight",
    "ANSWeight",
    "CompressedLinear",
    "convert_all_linear",
    "convert",
]


def mv_huffman(
    A_rem: Tensor, A_exp: Tensor, X: Tensor,
    offsets: Tensor,
    LUT1: Tensor, LUT2: Tensor, LUT3: Tensor, LUT4: Tensor,
    code_lengths: Tensor,
) -> Tensor:
    if A_rem.dim() == 2:
        A_rem = A_rem[None, :, :]
    if offsets.dim() == 2:
        offsets = offsets[None, :]
    
    split_k, M, N = A_rem.shape
    
    X = X.view(-1, split_k, N)
    
    assert M % 128 == 0
    assert N % 128 == 0
    assert len(X.shape) == 3
    assert X.size(-1) == N
    assert X.dtype == torch.bfloat16
    assert A_rem.device.type == A_exp.device.type == X.device.type == 'cuda'
    assert A_rem.device == A_exp.device == X.device
    
    A_rem = A_rem.contiguous()
    A_exp = A_exp.contiguous()
    X = X.contiguous()
    offsets = offsets.contiguous()
    LUT1 = LUT1.contiguous()
    LUT2 = LUT2.contiguous()
    LUT3 = LUT3.contiguous()
    LUT4 = LUT4.contiguous()
    code_lengths = code_lengths.contiguous()
    
    Y = torch.empty((X.size(0), M), dtype=torch.bfloat16, device=A_rem.device)
    gemv_bf16_huffman(A_rem, A_exp, X, Y, offsets, LUT1, LUT2, LUT3, LUT4, code_lengths)
    return Y


class HuffmanWeight(nn.Module):
    def __init__(
        self,
        rem: Tensor,
        exp: Tensor,
        offsets: Tensor,
        LUT1: Tensor,
        LUT2: Tensor,
        LUT3: Tensor,
        LUT4: Tensor,
        code_lengths: Tensor,
    ):
        super().__init__()
        self.rem = nn.Buffer(rem)
        self.exp = nn.Buffer(exp)
        self.offsets = nn.Buffer(offsets)
        self.LUT1 = nn.Buffer(LUT1)
        self.LUT2 = nn.Buffer(LUT2)
        self.LUT3 = nn.Buffer(LUT3)
        self.LUT4 = nn.Buffer(LUT4)
        self.code_lengths = nn.Buffer(code_lengths)
    
    def kwargs(self) -> dict[str, Tensor]:
        return dict(self.named_buffers())
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is not F.linear:
            return NotImplemented
        return linear_huffman(*args, **kwargs)
    
    def get_weight(self) -> Tensor:
        weight = torch.empty_like(self.rem, dtype=torch.bfloat16)
        weight = weight.view(weight.size(1), -1)
        assert weight.device == self.rem.device
        assert weight.device.type == 'cuda'
        huffman_decode(
            self.rem, self.exp, weight,
            self.offsets, self.LUT1, self.LUT2, self.LUT3, self.LUT4, self.code_lengths
        )
        return weight
    
    @classmethod
    def get_codec(cls, a: torch.Tensor) -> LUTHuffmanEncoder:
        device = torch.device('cuda', 0)
        _, exp = split_bf16(a.to(device))
        bincount = torch.bincount(exp.flatten(), minlength=256).to(torch.int64)
        codec = LUTHuffmanEncoder()
        codec.build_lut(bincount.cpu())
        return codec
    
    @classmethod
    def get_lut_kwargs(cls, codec: LUTHuffmanEncoder) -> dict[str, Tensor]:
        return {
            'LUT1': torch.from_numpy(codec.LUT1),
            'LUT2': torch.from_numpy(codec.LUT2),
            'LUT3': torch.from_numpy(codec.LUT3),
            'LUT4': torch.from_numpy(codec.LUT4),
            'code_lengths': torch.from_numpy(codec.code_lengths),
        }
    
    @classmethod
    def compress_groups(cls, x: Tensor, codec: LUTHuffmanEncoder) -> tuple[Tensor, Tensor]:
        device = x.device
        
        xx = x.flatten(0, 1) # [b x g, n]
        xx_output = torch.zeros(xx.size(0), xx.size(1) * 32, dtype=torch.uint8, device=device)
        xx_output.fill_(ord('0'))
        xx_output_lengths = torch.zeros(xx.size(0), dtype=torch.int32, device=device)
        codes = torch.tensor(
            [[ord(ch) for ch in codec.huffman_codes.get(i, '')[::-1].ljust(32, '\0')] for i in range(256)],
            dtype=torch.uint8, device=device
        )
        
        huffman_encode(xx, codes, xx_output, xx_output_lengths)
        
        xx_output_lengths = (xx_output_lengths + 8 - 1) // 8
        xx_output_lengths = ((xx_output_lengths.view(x.size(0), x.size(1)).max(dim=-1, 
            keepdim=True).values.expand(-1, x.size(1)) + 4 - 1) // 4 * 4).view_as(xx_output_lengths)
        xx_output_lengths = xx_output_lengths.view(x.size(0), x.size(1))[:, 0] // 4 # [b]
        
        xx_output = xx_output.view(xx_output.size(0), -1, 8)
        xx_output -= ord('0')
        # XXX: pytorch does not support int dot product, use f16 as a work around, with 10 bits mantissa, 2^10=1024
        bit_prod = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=device)
        xx_output = (xx_output.half() @ bit_prod.half()).to(torch.uint8)
        # xx_output = (xx_output * bit_prod).sum(dim=-1).to(torch.uint8)
        # [b x g, nj * 4]
        
        xx_output = xx_output.view(torch.int32) # [b x g, maxn]
        
        return xx_output, xx_output_lengths
    
    @classmethod
    def compress_part(cls, a: torch.Tensor, codec: LUTHuffmanEncoder) -> dict[str, Tensor]:
        assert a.size(1) % 128 == 0, a.shape
        
        device = torch.device('cuda', 0)
        
        rem, exp = split_bf16(a.to(device))
        
        x = exp
        x = x.view(rem.shape)
        x = x.reshape(x.size(0) // OP_PER_LANE, OP_PER_LANE, -1, 64, 2).transpose(1, 3).flatten(2, 4)
        
        xx_output, xx_output_lengths = cls.compress_groups(x, codec)

        xx_output = xx_output.view(x.size(0), x.size(1), -1).transpose(1, 2) # [b, maxn, g]
        xx_output = xx_output[torch.arange(xx_output.size(1))[None, :].to(device) < xx_output_lengths[:, None]] # [b, maxn]
        xx_output = torch.cat([xx_output, torch.zeros(1, xx_output.size(1), dtype=torch.int32).to(device)], dim=0)
        
        compressed_cuda = xx_output.flatten().cpu()
        offsets_cuda = torch.cat(
            [torch.tensor([0]).to(xx_output_lengths), xx_output_lengths.cumsum(dim=0).to(torch.int32) * x.size(1)]
        )[:-1].cpu()
        
        rem = rem.view(rem.size(0), -1, 2, 32, 2).transpose(-3, -2).reshape_as(rem)
        
        compressed = compressed_cuda
        offsets = offsets_cuda
        
        return {
            'rem': rem,
            'exp': compressed,
            'offsets': offsets,
        }
    
    @classmethod
    def compress_split_n(cls: type['HuffmanWeight'], a: torch.Tensor, n=2) -> 'HuffmanWeight':
        ori_size = a.shape
        codec = cls.get_codec(a)
        
        a = a.view(ori_size[0], n, -1).transpose(0, 1).contiguous()
        v = []
        for i in range(a.size(0)):
            # split the compress task into blocks to avoid OOM
            compress_bs = 4096
            split_m = (a.size(1) + compress_bs - 1) // compress_bs
            ss_a = [
                cls.compress_part(a[i][j*compress_bs:(j+1)*compress_bs], codec)
                for j in range(split_m)
            ]
            rem = torch.cat([x['rem'] for x in ss_a], dim=0)
            exp = torch.cat([x['exp'] for x in ss_a], dim=0)
            offsets = []
            count = 0
            for x in ss_a:
                offsets.append(x['offsets'] + count)
                count += x['exp'].nelement()
            offsets = torch.cat(offsets)
            sub_a = {
                'rem': rem,
                'exp': exp,
                'offsets': offsets,
            }
            v.append(sub_a)
        rem = torch.stack([sub_a['rem'] for sub_a in v], dim=0).cpu()
        exp = torch.cat([sub_a['exp'] for sub_a in v], dim=0).cpu()
        offsets = []
        count = 0
        for sub_a in v:
            offsets.append(sub_a['offsets'] + count)
            count += sub_a['exp'].nelement()
        offsets = torch.stack(offsets).cpu()
        
        return cls(
            rem=rem,
            exp=exp,
            offsets=offsets,
            **cls.get_lut_kwargs(codec),
        )
    
    @classmethod
    def compress(cls: type['HuffmanWeight'], a: torch.Tensor) -> 'HuffmanWeight':
        return cls.compress_split_n(a, n=get_block_size_n(a)[0])


class ANSWeight(HuffmanWeight):
    def __init__(
        self,
        rem: Tensor,
        exp: Tensor,
        offsets: Tensor,
        LUT: Tensor,
    ):
        nn.Module.__init__(self)
        self.rem = nn.Buffer(rem)
        self.exp = nn.Buffer(exp)
        self.offsets = nn.Buffer(offsets)
        self.LUT = nn.Buffer(torch.as_tensor(LUT))
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is not F.linear:
            return NotImplemented
        return linear_ans(*args, **kwargs)
    
    def get_weight(self) -> Tensor:
        weight = torch.empty_like(self.rem, dtype=torch.bfloat16)
        weight = weight.view(weight.size(1), -1)
        assert weight.device == self.rem.device
        assert weight.device.type == 'cuda'
        ans_decode(
            self.rem, self.exp, weight,
            self.offsets, self.LUT
        )
        return weight
    
    @classmethod
    def get_codec(cls, a: torch.Tensor) -> RANSEncoder:
        device = torch.device('cuda', 0)
        _, exp = split_bf16(a.to(device))
        bincount = torch.bincount(exp.flatten(), minlength=256).to(torch.int64)
        codec = RANSEncoder(bincount.tolist(), precision=10)
        return codec
    
    @classmethod
    def get_lut_kwargs(cls, codec: RANSEncoder) -> dict[str, Tensor]:
        lut = torch.tensor(codec.decode_lut, dtype=torch.int32)
        sym = lut[:, 0]
        freq = lut[:, 1]
        cum = lut[:, 2]
        lut = (sym << 24) | (freq << 12) | cum
        return {
            'LUT': lut,
        }
    
    @classmethod
    def compress_groups(cls, x: Tensor, codec: RANSEncoder) -> tuple[Tensor, Tensor]:
        device = x.device
        
        xx = x.flatten(0, 1) # [b x g, n]
        xx_output = torch.zeros(xx.size(0), xx.size(1) * 8, dtype=torch.int32, device=device) # [b x g, maxn]
        xx_output_lengths = torch.zeros(xx.size(0), dtype=torch.int32, device=device)
        freq = torch.tensor(codec.freq, dtype=torch.int32, device=device)
        cum = torch.tensor(codec.cum, dtype=torch.int32, device=device)
        ans_encode(xx, freq, cum, xx_output, xx_output_lengths)
        
        xx_output_lengths = xx_output_lengths.view(x.size(0), x.size(1)).max(dim=-1).values
        
        return xx_output, xx_output_lengths
    

class CompressedLinear(nn.Module):
    def __init__(self, weight: HuffmanWeight, bias: Optional[Tensor]) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


def convert(linear: nn.Linear, name, disable_precision_check=True, algo='huffman') -> CompressedLinear:
    assert algo in ('huffman', 'ans', 'smallest')
    
    if algo == 'huffman':
        weight = HuffmanWeight.compress(linear.weight)
    elif algo == 'ans':
        weight = ANSWeight.compress(linear.weight)
    elif algo == 'smallest':
        weight = min(
            [
                HuffmanWeight.compress(linear.weight),
                ANSWeight.compress(linear.weight),
            ],
            key=lambda w: sum(buffer.nelement() * buffer.element_size() for buffer in w.buffers())
        )

    module = CompressedLinear(
        weight=weight,
        bias=linear.bias,
    )
    
    # print_stats(linear.weight, module.weight)
    
    ori_size = linear.weight.nelement() * linear.weight.element_size()
    new_size = sum(buffer.nelement() * buffer.element_size() for buffer in module.weight.buffers())
    
    ratio = new_size / ori_size
    
    linear.cuda()
    module.cuda()
    x = torch.randn(1, linear.in_features, dtype=torch.bfloat16, device='cuda')
    with torch.no_grad():
        y1 = linear(x)
        y2 = module(x)
    max_rtol = ((y2 - y1).abs() / y1.abs().clamp(min=1e-4)).max().item()
    
    linear.cpu()
    module.cpu()
    
    fallback = False
    if ratio >= 1:
        print(f'\nWarning! {name} ratio = {ratio:.2%}')
        fallback = True
    # if not (y1 == y2).all():
    if not disable_precision_check and max_rtol >= 0.02: # the kernel the no longer deterministic after adding split_k block reduce
        print(f'\nWarning! {name} output mismatch: max rtol = {max_rtol:.2f}')
        fallback = True
    
    if fallback:
        print('Fallback to regular nn.Linear')
        return linear
    
    return module


def convert_all_linear(model: nn.Module, min_out_features=4096, progress=True, algo='huffman') -> None:
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.out_features >= min_out_features:
                targets.append(name)
    
    for name in tqdm(targets, disable=not progress):
        m = convert(model.get_submodule(name), name, algo=algo)
        model.set_submodule(name, m)


def linear_huffman(input: Tensor, weight: HuffmanWeight, bias: Optional[Tensor] = None) -> Tensor:
    assert input.dim() >= 2, input
    shape = input.shape
    input = input.flatten(0, -2)
    # assert input.size(0) == 1
    if input.size(0) <= 8:
        output = mv_huffman(
            weight.rem, weight.exp, input,
            weight.offsets,
            weight.LUT1, weight.LUT2, weight.LUT3, weight.LUT4,
            weight.code_lengths,
            # torch.cat([weight.LUT1, weight.LUT2, weight.LUT3, weight.LUT4, weight.code_lengths]),
        )
        if bias is not None:
            output += bias[None, :]
    else:
        weight = weight.get_weight()
        output = F.linear(input, weight, bias)
    
    output = output.unflatten(0, shape[:-1])
    
    return output


def mv_ans(
    A_rem: Tensor, A_exp: Tensor, X: Tensor,
    offsets: Tensor,
    LUT: Tensor,
) -> Tensor:
    if A_rem.dim() == 2:
        A_rem = A_rem[None, :, :]
    if offsets.dim() == 2:
        offsets = offsets[None, :]
    
    split_k, M, N = A_rem.shape
    
    X = X.view(-1, split_k, N)
    
    assert M % 128 == 0
    assert N % 128 == 0
    assert len(X.shape) == 3
    assert X.size(-1) == N
    assert X.dtype == torch.bfloat16
    assert A_rem.device.type == A_exp.device.type == X.device.type == 'cuda'
    assert A_rem.device == A_exp.device == X.device
    
    A_rem = A_rem.contiguous()
    A_exp = A_exp.contiguous()
    X = X.contiguous()
    offsets = offsets.contiguous()
    LUT = LUT.contiguous()
    
    Y = torch.empty((X.size(0), M), dtype=torch.bfloat16, device=A_rem.device)
    gemv_bf16_ans(A_rem, A_exp, X, Y, offsets, LUT)
    return Y


def linear_ans(input: Tensor, weight: ANSWeight, bias: Optional[Tensor] = None) -> Tensor:
    assert input.dim() >= 2, input
    shape = input.shape
    input = input.flatten(0, -2)
    # assert input.size(0) == 1
    if input.size(0) <= 8:
        output = mv_ans(
            weight.rem, weight.exp, input,
            weight.offsets,
            weight.LUT,
        )
        if bias is not None:
            output += bias[None, :]
    else:
        weight = weight.get_weight()
        output = F.linear(input, weight, bias)
    
    output = output.unflatten(0, shape[:-1])
    
    return output

