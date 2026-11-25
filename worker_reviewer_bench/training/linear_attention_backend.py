"""Use FLA's differentiable Gated DeltaNet kernel for this experiment."""
import inspect
import torch
from torch.utils.checkpoint import checkpoint
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling

reference_chunk_gated_delta_rule=inspect.unwrap(modeling.torch_chunk_gated_delta_rule)
HEADS_PER_CHUNK=2
NORM_ROWS_PER_CHUNK=262144
reference_gated_norm_forward=modeling.Qwen3_5MoeRMSNormGated.forward

def optimized_gated_norm(hidden_states,gate,weight,eps):
    """Preserve native BF16 rounding while bounding float32 norm workspace."""
    shape=hidden_states.shape
    hidden_states=hidden_states.reshape(-1,shape[-1])
    gate=gate.reshape(-1,shape[-1])
    assert not weight.requires_grad,'This experiment freezes normalization weights'
    def kernel(x,z,w):
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            outputs=[]
            for start in range(0,x.shape[0],NORM_ROWS_PER_CHUNK):
                states=x[start:start+NORM_ROWS_PER_CHUNK].float()
                variance=states.pow(2).mean(-1,keepdim=True)
                normalized=states*torch.rsqrt(variance+eps)
                # Match the native intermediate BF16 cast and weight product,
                # followed by FP32 SiLU gating and the final BF16 cast.
                normalized=w*normalized.to(x.dtype)
                normalized=normalized*torch.nn.functional.silu(z[start:start+NORM_ROWS_PER_CHUNK].float())
                outputs.append(normalized.to(x.dtype))
            return torch.cat(outputs,dim=0) if len(outputs)>1 else outputs[0]
    inputs=(hidden_states,gate,weight)
    if torch.is_grad_enabled() and any(t.requires_grad for t in inputs):
        output=checkpoint(kernel,*inputs,use_reentrant=True)
    else:
        output=kernel(*inputs)
    return output.reshape(shape)

def gated_norm_forward(self,hidden_states,gate=None):
    assert gate is not None and self.activation=='silu'
    return optimized_gated_norm(hidden_states,gate,self.weight,self.variance_epsilon)

def optimized_chunk_gated_delta_rule(query,key,value,g,beta,chunk_size=64,
        initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False,**kwargs):
    def kernel(q,k,v,gate,update):
        # Enter this context inside the callable so it is active during the
        # reentrant backward recomputation, not only the original forward.
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            assert q.shape[2]==k.shape[2]==v.shape[2]==gate.shape[2]==update.shape[2]
            outputs=[]
            final_states=[]
            # Recurrences are independent across heads. Retain the full time
            # axis in every call while bounding FLA's temporary state workspace.
            for start in range(0,v.shape[2],HEADS_PER_CHUNK):
                stop=min(start+HEADS_PER_CHUNK,v.shape[2])
                state=None if initial_state is None else initial_state[:,start:stop].contiguous()
                output,final_state=chunk_gated_delta_rule(
                    q[:,:,start:stop].contiguous(), k[:,:,start:stop].contiguous(),
                    v[:,:,start:stop].contiguous(), g=gate[:,:,start:stop].contiguous(),
                    beta=update[:,:,start:stop].contiguous(), initial_state=state,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
                outputs.append(output)
                if output_final_state:
                    final_states.append(final_state)
            return (torch.cat(outputs,dim=2) if len(outputs)>1 else outputs[0],
                    torch.cat(final_states,dim=1) if output_final_state else None)
    inputs=(query,key,value,g,beta)
    if (initial_state is None and not output_final_state and torch.is_grad_enabled()
            and any(t.requires_grad for t in inputs)):
        def output_only(*tensors):
            return kernel(*tensors)[0]
        # The kernel has flat tensor inputs, a single differentiable output,
        # and no parameters or mutable cache. Inner reentrant checkpointing
        # avoids retaining FLA internals in the outer nonreentrant layer's
        # recomputed-tensor cache. Trainer uses supported loss.backward().
        return checkpoint(output_only,*inputs,use_reentrant=True),None
    return kernel(*inputs)

def enable():
    modeling.torch_chunk_gated_delta_rule=optimized_chunk_gated_delta_rule
    modeling.Qwen3_5MoeRMSNormGated.forward=gated_norm_forward
    print('Gated DeltaNet backend: FLA 0.5.2 Triton kernel; no torch chunk fallback',flush=True)
    print('FLA kernel checkpoint: reentrant; saved activations offloaded during recomputation',flush=True)
    print(f'FLA workspace: {HEADS_PER_CHUNK} independent heads per call; complete sequence retained',flush=True)
    print(f'Gated RMS norm workspace: {NORM_ROWS_PER_CHUNK} rows per call; native rounding preserved',flush=True)
