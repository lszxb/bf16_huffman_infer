# bf16_huffman_infer

This is a experimental implementation of fused Decompression-GEMV kernel, using the LUT-based Huffman compression purposed by [DFloat11](https://github.com/LeanModels/DFloat11), to compress the exponential bits of the BF16 format. It provides reduced memory usage of the LLMs, while maintaining comparable decoding speed.

The current fused kernel implementation only support `batch_size<=8`, otherwise it will fallback to the non-fused decompression then GEMM implementation. Due to the optimized data layout, it can achieve about 80%~90% decoding speed of the original model, while reducing the VRAM usage by ~25%. The compression ratio is slightly higher than the original DFloat11, but the decoding speed is much faster. On some bandwidth-limited GPUs, like RTX-4060ti, it can even achieve better decoding speed than the original BF16 model.


## Installation

```bash
pip install bf16_huffman_infer
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
