from time import monotonic
import torch
from bf16_huffman_infer.cudagraph_utils import get_graphed_model
from bf16_huffman_infer.utils import get_model_size
from bf16_huffman_infer.functional import convert_all_linear
from transformers import (
    AutoModelForCausalLM, Qwen2ForCausalLM, AutoTokenizer, TextStreamer, StaticCache,
)

def main():
    path = 'Qwen/Qwen3-8B'
    model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype='auto',
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(path)

    prompt = '"Hello, world!" is a common phrase used in programming examples. It'
    inputs = tok(prompt, return_tensors='pt')
    inputs = inputs.to('cuda')
    
    ori_size = get_model_size(model)
    gen_len = 256

    cache = StaticCache(
        model.config, max_batch_size=1, max_cache_len=1024,
        device='cuda', dtype=model.config.torch_dtype,
    )


    # Native BF16 inference
    model.cuda()
    graphed_model = get_graphed_model(model, cache)

    for _ in range(3):
        start = monotonic()
        graphed_model.generate(
            **inputs, streamer=TextStreamer(tok),
            max_new_tokens=gen_len, min_new_tokens=gen_len,
        )
        torch.cuda.synchronize()
        print(f'Native BF16 inference takes {monotonic() - start:.2f}s')

    del graphed_model
    model.cpu()


    # Perform BF16 Huffman compression
    convert_all_linear(model.model, min_out_features=0)
    new_size = get_model_size(model)
    overall_compression_ratio = new_size / ori_size
    print(f'ori_size = {ori_size / 1024 ** 3:.2f}GiB')
    print(f'new_size = {new_size / 1024 ** 3:.2f}GiB')
    print(f'{overall_compression_ratio = :.2%}')


    # Compressed BF16 inference
    model.cuda()
    graphed_model = get_graphed_model(model, cache)

    for _ in range(3):
        start = monotonic()
        graphed_model.generate(
            **inputs, streamer=TextStreamer(tok),
            max_new_tokens=gen_len, min_new_tokens=gen_len,
        )
        torch.cuda.synchronize()
        print(f'Compressed BF16 inference takes {monotonic() - start:.2f}s')


if __name__ == '__main__':
    main()
