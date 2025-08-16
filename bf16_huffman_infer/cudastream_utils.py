import torch
from torch import nn, Tensor
from transformers import PreTrainedModel
from .utils import shallow_copy_model


def parallelize_qkv(self_attn):
    main_stream = None
    q_stream = None
    k_stream = None
    
    def q_hook(m: nn.Module, inp: tuple[Tensor]):
        nonlocal main_stream, q_stream
        
        x = inp[0]
        
        main_stream = torch.cuda.current_stream(x.device)
        q_stream = torch.cuda.Stream(device=x.device)
        q_stream.wait_stream(main_stream)
        torch.cuda.set_stream(q_stream)
    
    def k_hook(m: nn.Module, inp: tuple[Tensor]):
        nonlocal k_stream
        
        x = inp[0]
        
        k_stream = torch.cuda.Stream(device=x.device)
        k_stream.wait_stream(main_stream)
        torch.cuda.set_stream(k_stream)
    
    def qk_post_hook(m: nn.Module, inp: tuple[Tensor], out: Tensor):
        out.record_stream(main_stream)
    
    def v_hook(m: nn.Module, inp: tuple[Tensor]):
        torch.cuda.set_stream(main_stream)
    
    def v_post_hook(m: nn.Module, inp: tuple[Tensor], out: Tensor):
        main_stream.wait_stream(q_stream)
        main_stream.wait_stream(k_stream)
    
    
    self_attn.q_proj.register_forward_pre_hook(q_hook)
    self_attn.k_proj.register_forward_pre_hook(k_hook)
    self_attn.v_proj.register_forward_pre_hook(v_hook)
    self_attn.q_proj.register_forward_hook(qk_post_hook)
    self_attn.k_proj.register_forward_hook(qk_post_hook)
    self_attn.v_proj.register_forward_hook(v_post_hook)


def parallelize_qkv_model(model: PreTrainedModel):
    for layer in model.model.layers:
        parallelize_qkv(layer.self_attn)


def get_parallelize_qkv_model(model: PreTrainedModel) -> PreTrainedModel:
    new_model = shallow_copy_model(model)
    parallelize_qkv_model(new_model)
    return new_model
