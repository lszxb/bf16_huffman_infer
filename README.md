# bf16_huffman_infer

This is a experimental implementation of fused Huffman-decomposition-GEMV kernel, using the LUT-based Huffman compression purposed by [DFloat11](https://github.com/LeanModels/DFloat11), to compress the exponential bits of the BF16 format.

It provides reduced memory usage of the LLMs, while maintaining comparable decoding speed under `batch_size=1`.

It's in very early stage and should only work in very limited scenarios.


## Usage

```python
import torch
from bf16_huffman_infer import (
    get_graphed_model, get_model_size, convert_all_linear,
)
from transformers import (
    AutoModelForCausalLM, Qwen3ForCausalLM, AutoTokenizer, TextStreamer, StaticCache,
)

path = 'Qwen/Qwen3-8B'
model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype='auto', low_cpu_mem_usage=True,
)
tok = AutoTokenizer.from_pretrained(path)
config = model.config

# currently only batch_size=1 is supported
inputs = tok('"Hello, world!" is', return_tensors='pt')

ori_size = get_model_size(model)

# compress the model, will use cuda:0 for computation, can be done in a few minutes
convert_all_linear(model.model, min_out_features=0)
torch.cuda.empty_cache()

new_size = get_model_size(model)
overall_compression_ratio = new_size / ori_size
print(f'ori_size = {ori_size / 1024 ** 3:.2f}GiB')
print(f'new_size = {new_size / 1024 ** 3:.2f}GiB')
print(f'{overall_compression_ratio = :.2%}')

model.cuda()

# Optional, but necessary to get maximize decoding latency for small models
graphed_model = get_graphed_model(
    model,
    StaticCache(
        config, max_batch_size=1, max_cache_len=1024,
        device=model.device, dtype=config.torch_dtype,
    )
)
graphed_model.generate(
    **inputs.to(model.device), streamer=TextStreamer(tok), max_new_tokens=128,
)
```
