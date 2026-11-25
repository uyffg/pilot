"""Preserve LoRA dropout masks during FSDP activation recomputation."""
import torch
import importlib
wrappers=importlib.import_module('torch.distributed.algorithms._checkpoint.checkpoint_wrapper')

def verify_dropout_gradients():
    # Include a negative control so the check detects the installed policy's
    # different recomputation mask even when checkpoint inputs are frozen.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(38)
        inputs=torch.randn(32,16)
        model=torch.nn.Sequential(torch.nn.Dropout(0.05),torch.nn.Linear(16,16))
        torch.manual_seed(438)
        reference=model(inputs)
        reference.sum().backward()
        gradients=[p.grad.clone() for p in model.parameters()]
        for preserve in (False,True):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(438)
            wrapped=wrappers.checkpoint_wrapper(model,preserve_rng_state=preserve)
            output=wrapped(inputs)
            torch.testing.assert_close(output,reference,rtol=0,atol=0)
            output.sum().backward()
            differs=any(not torch.equal(p.grad,expected) for p,expected in zip(model.parameters(),gradients))
            assert differs is (not preserve),('Dropout gradient preservation',preserve)

def enable():
    verify_dropout_gradients()
    import accelerate.accelerator as accelerator_module
    import accelerate.utils.fsdp_utils as fsdp_utils
    original_apply=fsdp_utils.fsdp2_apply_ac
    def apply_with_preserved_rng(accelerator,model):
        original_wrapper=wrappers.checkpoint_wrapper
        count=0
        def preserving_wrapper(*args,**kwargs):
            nonlocal count
            kwargs['preserve_rng_state']=True
            count+=1
            return original_wrapper(*args,**kwargs)
        wrappers.checkpoint_wrapper=preserving_wrapper
        try:
            result=original_apply(accelerator,model)
        finally:
            wrappers.checkpoint_wrapper=original_wrapper
        assert count>0,'Expected FSDP activation checkpoint wrappers'
        print(f'FSDP activation checkpoint RNG preserved: {count} modules',flush=True)
        return result
    fsdp_utils.fsdp2_apply_ac=apply_with_preserved_rng
    accelerator_module.fsdp2_apply_ac=apply_with_preserved_rng
    print('Checkpoint dropout-gradient parity passed; negative control detected changed masks',flush=True)

if __name__=='__main__':
    verify_dropout_gradients()
    print('Dropout checkpoint output/gradient parity passed; negative control failed as expected',flush=True)
