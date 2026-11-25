"""Make the frozen output projection usable by Liger with FSDP2."""
import torch
from torch.distributed.tensor import DTensor
from liger_kernel.transformers.model import qwen3_5_moe as liger_model

original_loss=liger_model.LigerForCausalLMLoss
frozen_head=None

def compatible_loss(*args,**kwargs):
    global frozen_head
    weight=kwargs.get('lm_head_weight')
    hidden=kwargs.get('hidden_states')
    if isinstance(weight,DTensor):
        assert not weight.requires_grad, 'This gather is for the frozen LoRA output head only'
        if frozen_head is None:
            # CPU-offloaded DTensors must move their local shards to CUDA
            # before NCCL can gather them. The frozen projection never changes.
            frozen_head=weight.to(device=hidden.device).full_tensor().detach()
            assert not isinstance(frozen_head,DTensor)
            print('FSDP2 fused loss: gathered frozen output projection',tuple(frozen_head.shape),flush=True)
        assert frozen_head.shape==weight.shape and frozen_head.device==hidden.device
        kwargs['lm_head_weight']=frozen_head
    return original_loss(*args,**kwargs)

def enable():
    liger_model.LigerForCausalLMLoss=compatible_loss
    print('FSDP2 fused loss adapter enabled for frozen language output projection',flush=True)
