# bf16_huffman_infer

This is a experimental implementation of fused Huffman-decomposition-GEMV kernel, using the LUT-based Huffman compression purposed by [DFloat11](https://github.com/LeanModels/DFloat11), to compress the exponential bits of the BF16 format.

It provides reduced memory usage of the LLMs, while maintaining comparable decoding speed under `batch_size=1`.

It's in very early stage and should only work in very limited scenarios.
