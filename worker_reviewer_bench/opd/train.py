"""One full-batch OPD update per freshly sampled rollout round, Qwen3.5 FSDP2."""
import argparse
import json
import math
import os
from pathlib import Path
import types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--adapter')
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--max-len', type=int, default=131072)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--chunk-size', type=int, default=16)
    args = p.parse_args()
    # Reuse the established rank-zero loader, meta initialization, FSDP2 CPU
    # offload, differentiable FLA kernels, RNG and finite-metrics checks.
    from ..training import qwen35 as infrastructure
    import torch
    import transformers
    from accelerate import PartialState
    from peft import PeftModel, LoraConfig, get_peft_model
    from torch.distributed.tensor import DTensor
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from .data import encode_pair
    from .loss import full_vocab_kl

    transformers.set_seed(args.seed)
    rows = [json.loads(line) for line in Path(args.pairs).read_text().splitlines() if line.strip()]
    if not rows or len({r['student_revision'] for r in rows}) != 1 or len({r['round'] for r in rows}) != 1:
        raise ValueError('Expected one nonempty on-policy round from a single student revision')
    tok = transformers.AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True)
    dataset = [encode_pair(row, tok, args.max_len) for row in rows]
    model = transformers.AutoModelForCausalLM.from_pretrained(os.environ['MODEL_PATH'], dtype=torch.bfloat16)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
            lora_dropout=0.0, target_modules='all-linear', task_type='CAUSAL_LM'))
    head = model.get_output_embeddings()
    if head.weight.requires_grad:
        raise ValueError('Output head must remain frozen')
    # The unchanged causal-LM backbone returns its last hidden states in the
    # logits slot, without ever allocating full-context vocabulary logits.
    def hidden_only(self, hidden_states):
        return hidden_states
    head.forward = types.MethodType(hidden_only, head)

    class OPDTrainer(transformers.Trainer):
        projection = None
        observed_losses = []

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # LoRA dropout is disabled for identical teacher/student weights.
            # Eval mode does NOT disable gradients through the student.
            model.eval()
            device = self.accelerator.device
            target_len = len(inputs['target_ids'])
            teacher = inputs['teacher']
            student = inputs['student']
            with torch.no_grad():
                teacher_output = model(input_ids=torch.tensor([teacher['input_ids']], device=device),
                                       use_cache=False)
                th = teacher_output.logits[0, teacher['start']:teacher['start'] + target_len].detach().clone()
                del teacher_output
            output = model(input_ids=torch.tensor([student['input_ids']], device=device), use_cache=False)
            sh = output.logits[0, student['start']:student['start'] + target_len]
            if self.projection is None:
                weight = head.weight
                self.projection = (weight.to(device).full_tensor().detach()
                                   if isinstance(weight, DTensor) else weight.detach().to(device))
            loss = full_vocab_kl(sh, th, self.projection, args.chunk_size)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite OPD KL')
            self.observed_losses.append(float(loss.detach()))
            return (loss, output) if return_outputs else loss

    world = int(os.environ.get('WORLD_SIZE', '1'))
    # All examples are evaluated at the SAME parameters before this one
    # optimizer step. DistributedSampler may repeat up to world-1 examples.
    accumulation = math.ceil(len(dataset) / world)
    targs = transformers.TrainingArguments(output_dir=args.out, num_train_epochs=1,
        max_steps=1, learning_rate=args.lr, lr_scheduler_type='constant', warmup_steps=0,
        per_device_train_batch_size=1, gradient_accumulation_steps=accumulation,
        bf16=True, eval_strategy='no', save_strategy='no', report_to=[],
        remove_unused_columns=False, seed=args.seed, use_liger_kernel=False)
    trainer = OPDTrainer(model=model, args=targs, train_dataset=dataset,
                         data_collator=lambda batch: batch[0])
    # A custom loss already averages action tokens; Trainer should average
    # microbatches instead of trying to normalize with absent CE labels.
    trainer.model_accepts_loss_kwargs = False
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autograd.graph.save_on_cpu(pin_memory=False):
        trainer.train()
    if trainer.state.global_step != 1:
        raise RuntimeError('Each on-policy round must perform exactly one update')
    statistics = torch.tensor([sum(trainer.observed_losses), len(trainer.observed_losses)],
                              dtype=torch.float64, device=trainer.accelerator.device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(statistics)
    mean_kl = float((statistics[0] / statistics[1]).item())
    trainer.accelerator.wait_for_everyone()
    adapter_state = {}
    for name, value in sorted(trainer.model.state_dict().items()):
        if 'lora_' not in name and 'modules_to_save' not in name:
            continue
        tensor = value.full_tensor() if isinstance(value, DTensor) else value
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f'Nonfinite adapter {name}')
        if trainer.is_world_process_zero():
            adapter_state[name] = tensor.detach().cpu().clone()
    if trainer.is_world_process_zero():
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out, state_dict=adapter_state)
        tok.save_pretrained(out)
        trainer.state.save_to_json(str(out / 'trainer_state.json'))
        update_path = out / 'opd_update.json.tmp'
        update_path.write_text(json.dumps({
            'objective': 'full_vocabulary_KL_teacher_to_student', 'optimizer_steps': 1,
            'pairs': len(rows), 'student_revision': rows[0]['student_revision'],
            'round': rows[0]['round'], 'world_size': world,
            'distributed_padding_pairs': accumulation * world - len(rows),
            'mean_kl': mean_kl,
            'rank0_mean_kl': sum(trainer.observed_losses) / len(trainer.observed_losses),
            'lr': args.lr, 'max_len': args.max_len, 'reviewer_in_student_history': False,
            'previous_adapter': args.adapter}, indent=2))
        update_path.replace(out / 'opd_update.json')
    trainer.accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
