# bf16_huffman_infer

This is a experimental implementation of fused Decompression-GEMV kernel, using the LUT-based Huffman compression purposed by [DFloat11](https://github.com/LeanModels/DFloat11), to compress the exponential bits of the BF16 format. It provides reduced memory usage of the LLMs, while maintaining comparable decoding speed to the regular BF16 format.

The current fused kernel implementation only support `batch_size<=8`, otherwise it will fallback to the non-fused decompression then GEMM implementation. Due to the optimized data layout, it can achieve about 80%~90% decoding speed of the original model, while reducing the VRAM usage by ~25%. The compression ratio is slightly higher than the original DFloat11, but the decoding speed is much faster. On some bandwidth-limited GPUs, like RTX-4060ti, it can even achieve better decoding speed than the original BF16 model.


## Benchmark Results

The following is the time used to generate 256 tokens with batch size 1 on different GPUs, using the script `examples/benchmark.py`. Please note that the CUDA Graph is used during the benchmark to minimize the CPU kernel launch overhead.

| Model      | Device     | Raw BF16 Time | Compressed BF16 Time | Raw / Compressed Size |
| ---------- | ---------- | ------------- | -------------------- | --------------------- |
| Qwen2.5 7B | RTX 4060Ti | 14.98s        | 13.02s               | 14.19 / 10.99 GiB     |
|            | RTX A6000  | 6.66s         | 7.23s                |                       |
| Qwen3 8B   | RTX 4060Ti | OOM           | 14.11s               | 15.26 / 11.52 GiB     |
|            | RTX A6000  | 7.75s         | 8.24s                |                       |


## Installation

You can directly install the package from pypi, which will compile the custom CUDA extension during installation.
```bash
pip install --no-build-isolation bf16_huffman_infer
```
or you can also clone the repo and install it manually:
```bash
git clone https://github.com/lszxb/bf16_huffman_infer.git
cd bf16_huffman_infer
pip install --no-build-isolation -e .
```


## Requirements
- Python 3.9+
- PyTorch 2.7+
- Nvidia Turing or newer GPU


## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, StaticCache
from bf16_huffman_infer import get_graphed_model, convert_all_linear

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B', torch_dtype='auto')
tok = AutoTokenizer.from_pretrained(path)

# currently only batch_size<=8 is supported
inputs = tok('"Hello, world!" is', return_tensors='pt')

# a single line to compress the model
# will use cuda:0 for computation, can be done in a few minutes
convert_all_linear(model.model, min_out_features=0)
model.cuda()

# graphed_model = model
# Optional, but necessary to get maximize decoding latency for small models
graphed_model = get_graphed_model(
    model,
    StaticCache(
        model.config, max_batch_size=1, max_cache_len=1024,
        device=model.device, dtype=model.config.torch_dtype,
    )
)
graphed_model.generate(
    **inputs.to(model.device), streamer=TextStreamer(tok), max_new_tokens=128,
)
```
