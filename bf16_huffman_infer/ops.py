import torch
from torch import Tensor
from . import _C

    
def gemv_bf16_huffman(
    A_rem: Tensor, A_exp: Tensor, X: Tensor, Y: Tensor,
    offsets: Tensor,
    LUT1: Tensor, LUT2: Tensor, LUT3: Tensor, LUT4: Tensor,
    code_lengths: Tensor,
) -> None:
    torch.ops.bf16_huffman_infer.gemv_bf16_huffman.default(
        A_rem, A_exp, X, Y,
        offsets, LUT1, LUT2, LUT3, LUT4, code_lengths
    )
    
    
def huffman_encode(
    data: Tensor, LUT: Tensor, output: Tensor, output_lengths: Tensor
) -> None:
    torch.ops.bf16_huffman_infer.huffman_encode.default(data, LUT, output, output_lengths)
    
    
def huffman_decode(
    A_rem: Tensor, A_exp: Tensor, Y: Tensor,
    offsets: Tensor,
    LUT1: Tensor, LUT2: Tensor, LUT3: Tensor, LUT4: Tensor,
    code_lengths: Tensor,
) -> None:
    torch.ops.bf16_huffman_infer.huffman_decode.default(
        A_rem, A_exp, Y,
        offsets, LUT1, LUT2, LUT3, LUT4, code_lengths
    )

