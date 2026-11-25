"""Qwen3.5 LoRA FSDP2 setup for OPD."""
print('TRAINER_PROCESS_START', flush=True)
import inspect
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, TrainingArguments
import transformers
from transformers import TrainerCallback
import math
import peft
from accelerate import PartialState, init_empty_weights
import torch
from .linear_attention_backend import enable as enable_linear_attention
enable_linear_attention()
from .fsdp_fused_loss import enable as enable_fsdp_fused_loss
enable_fsdp_fused_loss()
from .checkpoint_rng_policy import enable as enable_checkpoint_rng_policy
enable_checkpoint_rng_policy()

# Initialize torchrun ranks before from_pretrained so FSDP's RAM-efficient
# loader loads checkpoint weights on rank zero rather than eight full copies.
if int(os.environ.get('WORLD_SIZE','1')) > 1:
    PartialState(backend='cpu:gloo,cuda:nccl',timeout=timedelta(hours=2))

config=AutoConfig.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True)
text_config=config.get_text_config()
text_config.output_router_logits=False
config.use_cache=False
text_config.use_cache=False
decoder_class = 'Qwen3NextDecoderLayer' if 'qwen3_next' in text_config.model_type else (
    'Qwen3_5MoeDecoderLayer' if 'qwen3_5_moe' in text_config.model_type else 'Qwen3MoeDecoderLayer')
from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
assert config.model_type in MODEL_TYPE_TO_APPLY_LIGER_FN, (
    f'Installed Liger does not support {config.model_type}; upgrade the isolated environment before training')

original_load = AutoModelForImageTextToText.from_pretrained
original_args = TrainingArguments
original_lora_config = peft.LoraConfig

class FiniteMetrics(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ('loss', 'grad_norm', 'train_loss'):
            if logs is not None and key in logs:
                if not math.isfinite(float(logs[key])):
                    raise FloatingPointError(f'Non-finite {key} at optimizer step {state.global_step}')

original_trainer = transformers.Trainer
class CheckedTrainer(original_trainer):
    def __init__(self, *args, **kwargs):
        kwargs['callbacks'] = list(kwargs.get('callbacks') or []) + [FiniteMetrics()]
        super().__init__(*args, **kwargs)

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            logs['gpu_peak_gib'] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        return super().log(logs, *args, **kwargs)


transformers.Trainer = CheckedTrainer

def language_lora_config(*args, **kwargs):
    # Preserve the frozen vision tower even when PEFT selects all linear layers.
    kwargs['exclude_modules']=r'.*visual.*'
    return original_lora_config(*args, **kwargs)

peft.LoraConfig=language_lora_config


def load_model(*args,**kwargs):
    kwargs.update(config=config,attn_implementation='sdpa',local_files_only=True)
    if torch.distributed.is_initialized() and torch.distributed.get_rank()!=0:
        # FSDP2 distributes rank zero's full state after sharding on meta.
        # Avoid both checkpoint copies and full CPU zero tensors on other ranks.
        # Keep non-persistent buffers real so rotary frequencies remain valid.
        with init_empty_weights(include_buffers=False):
            model=AutoModelForImageTextToText.from_config(config,
                dtype=kwargs.get('dtype',kwargs.get('torch_dtype',torch.float32)),
                attn_implementation='sdpa')
        assert all(p.device.type=='meta' for p in model.parameters())
        assert all(b.device.type!='meta' for b in model.buffers())
        print('FSDP2 loading: nonzero rank uses meta parameters with real buffers',flush=True)
    else:
        model=original_load(*args,**kwargs)
        print('FSDP2 loading: rank zero loaded pretrained weights',flush=True)
    # LoRA is text-only: preserve the pretrained vision tower.
    if hasattr(model.model,'visual'):
        model.model.visual.requires_grad_(False)
    actual={type(layer).__name__ for layer in model.modules() if 'DecoderLayer' in type(layer).__name__}
    assert decoder_class in actual, (decoder_class, actual)
    print('Qwen3.5 LoRA decoder:',decoder_class,'model:',type(model).__name__,flush=True)
    return model


def configured_arguments_init(self,*args,**kwargs):
    # Keep the original signature visible: the established trainer inspects it
    # for warmup and fused-CE support.
    kwargs['fsdp']='full_shard auto_wrap' + (' offload' if bool(int(os.environ.get('TRAIN_CPU_OFFLOAD','1'))) else '')
    kwargs['fsdp_config']={'fsdp_version':2,'transformer_layer_cls_to_wrap':[decoder_class],
                           'cpu_ram_efficient_loading':True,'sync_module_states':True,
                           'use_orig_params':True,'limit_all_gathers':True,
                           'cpu_offload':bool(int(os.environ.get('TRAIN_CPU_OFFLOAD','1'))),
                           'reshard_after_forward':True}
    kwargs['gradient_checkpointing_kwargs']={'use_reentrant':False}
    kwargs['gradient_checkpointing']=False
    kwargs['fsdp_config']['activation_checkpointing']=True
    kwargs['logging_steps']=1
    kwargs['logging_nan_inf_filter']=False
    kwargs['dataloader_num_workers']=0
    kwargs['accelerator_config']={'gradient_accumulation_kwargs':{'sync_each_batch':True}}
    kwargs['optim']='adamw_torch'
    kwargs['ddp_timeout']=7200
    original_args.__init__(self,*args,**kwargs)


configured_arguments_init.__signature__=inspect.signature(original_args.__init__)
class arguments(original_args):
    __init__=configured_arguments_init

arguments.__signature__=inspect.signature(original_args)
AutoModelForCausalLM.from_pretrained=load_model
transformers.TrainingArguments=arguments
