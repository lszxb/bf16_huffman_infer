from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel, StaticCache, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import copy


def apply_graphed(model: PreTrainedModel, cache: StaticCache, replace_generate=True):
    _old_forward = model.forward
    
    def _forward(
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        cache_position: Optional[torch.LongTensor],
    ):
        with torch.no_grad():
            return _old_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=True,
                cache_position=cache_position,
                return_dict=True,
            ).logits
    
    cache.requires_grad = False
    
    
    dtype = model.config.torch_dtype
    device = model.device
    
    _forward_graphed = torch.cuda.make_graphed_callables(
        _forward,
        sample_args=(
            torch.zeros(1, 1, dtype=torch.int64, device=device),
            # torch.zeros(1, 1, 1, cache.get_max_cache_shape(), dtype=dtype, device=device),
            torch.zeros(1, 1, 1, cache.get_max_cache_shape(), dtype=dtype, device=device),
            torch.zeros(1, 1, dtype=torch.int64, device=device),
            # this is the cache position, set to last position to avoid modifying the existing cache
            torch.tensor([cache.get_max_cache_shape() - 1], dtype=torch.int64, device=device),
        ),
    )
    cache.reset()
    
    def forward(
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[StaticCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if input_ids.shape[1] == 1:
            func = _forward_graphed
        else:
            func = _forward
        
        logits = func(input_ids, attention_mask, position_ids, cache_position)
        return CausalLMOutputWithPast(logits=logits, past_key_values=cache)
    
    model.forward = forward
    
    if replace_generate:
        _old_generate = model.generate
        model.generate = lambda *args, **kwargs: cache.reset() or _old_generate(
            *args, past_key_values=cache, disable_compile=True, **kwargs,
        )


def shallow_copy_model(model: nn.Module) -> nn.Module:
    memo = {}
    for p in model.parameters():
        memo[id(p)] = p
    for b in model.buffers():
        memo[id(b)] = b
    return copy.deepcopy(model, memo=memo)


def get_graphed_model(model: PreTrainedModel, cache: StaticCache) -> PreTrainedModel:
    graphed_model = shallow_copy_model(model)
    apply_graphed(graphed_model, cache)
    return graphed_model
