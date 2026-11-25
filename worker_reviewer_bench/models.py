"""Shared model merge and serving commands, independent of experiment paths."""
import shutil
from pathlib import Path
import subprocess
import sys


def merge(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Merged model output directory must be empty')
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16,
        device_map={'': 'cpu'}, local_files_only=True, attn_implementation='sdpa')
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False, local_files_only=True)
    for name, parameter in model.named_parameters():
        if 'lora_' in name and not torch.isfinite(parameter).all().item():
            raise ValueError(f'Nonfinite adapter tensor: {name}')
    model = model.merge_and_unload(safe_merge=True)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, max_shard_size='5GB', safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model, local_files_only=True).save_pretrained(output)
    for name in ['preprocessor_config.json', 'processor_config.json', 'video_preprocessor_config.json']:
        source = Path(args.model) / name
        if source.exists():
            shutil.copy2(source, output/name)
    return 0


def serve(args):
    return subprocess.call([sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', args.model,
        '--served-model-name', args.name, '--host', args.host, '--port', str(args.port),
        '--dtype', 'bfloat16', '--tensor-parallel-size', str(args.gpus),
        '--distributed-executor-backend', 'mp', '--max-model-len', str(args.max_len),
        '--max-num-seqs', str(args.max_seqs), '--gpu-memory-utilization', str(args.gpu_memory),
        '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder',
        '--language-model-only', '--enable-prefix-caching', '--enforce-eager', '--seed', str(args.seed)])
