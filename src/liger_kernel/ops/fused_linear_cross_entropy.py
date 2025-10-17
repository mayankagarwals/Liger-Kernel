import torch
import triton

from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.utils import amp_custom_bwd
from liger_kernel.ops.utils import amp_custom_fwd
from liger_kernel.ops.utils import element_mul_kernel
from liger_kernel.ops.utils import is_hip

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2


    '''
    Example 1: 
       
        _input: hidden_states right after the model forward: [batch, seq_len, d_model]; after reshape(-1, d_model) it becomes [(batch * seq_len), d_model]
        weight: model.lm_head.weight: [vocab_size, d_model] (PyTorch stores linear weights as [out_features, in_features])
        target: targets from target_ids.reshape(-1): length batch * seq_len.

    '''

def fused_linear_cross_entropy_forward(
    _input,
    weight,
    target,
    ce_weight=None,
    bias=None,
    ignore_index=-100,
    lse_square_scale=0.0,
    label_smoothing=0.0, # for llms, this is almost almost 0.0 so you can assume the same.
    reduction="mean",
    softcap=None,
    return_z_loss=False,
):
    assert isinstance(return_z_loss, bool), f"return_z_loss must be True or False. Got: {return_z_loss}"
    device = _input.device

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048

    '''
    For GPT2 
    V (vocab size): 50,257
    T (context length): 1,024
    H (hidden size / n_embd):GPT-2 117M (small): 768

    Batch size depends on your parallelism setup. 
    117M params means you want to train on ~2.3B tokens
    for global batch of 0.52M tokens per step  (4,387 steps to reach optimality). Why is this chosen? Lot of stuff goes here, let's not get into that 
    This is 512 global sequences sequences (0.5M/1k)

    8 GPUs, 8 gradient accumulation factor 
    8 sequence per gpu . 8x8x8 leads to 512

    '''

    BT, H = _input.shape #  Note how this is running on a single gpu.  So batch size is 8. [8192, 768]
    V = weight.shape[0] # Note how pytorch stores linear weights as [out_features, in_features]. Here it is [V, d_model] [50257, 768]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V)) #  min( 65536 // 2,  65536 ) ->  65536 // 2

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H. # cdiv(50257, 768) -> 67. This is how many chunks of H will make V 
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))  # (BT + inc_factor - 1) // inc_factor. we know cdiv(BT, inc_factor) will allow us to have BTchunked x V at lower size than BT x H . However, we would like BT chunked to be power of 2 . Wait to find out why . Here it is 122 . Next power of 2 is 128
    num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size. With our found chunk size, how many chunks do we need  -> 64 chunks 

    grad_weight = torch.zeros_like(weight, device=device) if weight.requires_grad else None # example 1 : weight.requires grad will be true as it is the lm_head layer. 
    grad_input = torch.zeros_like(_input, device=device)
    grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None # example 1: bias is none

    # we use fp32 for loss accumulator
    loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)
    z_loss_1d = torch.zeros(BT, dtype=_input.dtype, device=_input.device) if return_z_loss else None

    # TODO: evaluate how CUDA synchronization caused by .item() affects the speed
    target_mask = target != ignore_index # Simple mask that is also of shape BXT. It has false if it is a padding index. 
    '''
    Why padding? 
    Padding let's us pack variable length sequences into fixed [batch,token] tensors. It's useful for training because if we trained for a fixed sequece length , it doesn't mimic real world behaviour of short to large sequence lengths. Can introduce some bias. not sure what 
    though , can learn more here.
    '''

    total_n_non_ignore = target_mask.sum().item()
    total_sum_non_ignore_ce_weight = total_n_non_ignore
    ce_weight_sum = 0.0

    '''

    Deep dive into ce_weight or cross entropy weight in ce_weight.md 
    For LLMs, we ideally want the class imbalance also to be learnt so we don't use it 

    '''

    if ce_weight is not None:
        assert ce_weight.shape[0] == V, f"If given, weight has to be a Tensor of size V. Got: {ce_weight.shape}"
        assert torch.is_floating_point(ce_weight), (
            f"If given, weight has to be a Tensor of floating point dtype. Got: {ce_weight.dtype}"
        )
        total_sum_non_ignore_ce_weight = (
            torch.gather(ce_weight, dim=0, index=target.masked_select(target_mask)).sum().item()
        )
        ce_weight_sum = ce_weight.sum().item()
        if ce_weight.stride(-1) != 1:
            ce_weight = ce_weight.contiguous()

        '''

        Deep dive into .contiguous 

        Imagine you start with a 1-D class-weight tensor ce_weight that is already contiguous:

        ce_weight = torch.tensor([0.1, 0.4, 0.5])

        Now slice every other entry: ce_weight = ce_weight[::2]. PyTorch doesn’t copy data here; it returns a view that jumps through memory with stride 2. Many kernels (including the Triton kernel we call) expect the last dimension to have stride 1—i.e., adjacent elements in memory—so they can iterate efficiently and without stride logic. Hitting such a kernel with the strided view can produce wrong results or runtime errors.

        Calling ce_weight = ce_weight.contiguous() forces PyTorch to copy the view into a new, properly packed buffer (stride 1). Subsequent GPU kernels then see a dense vector, match expectations, and run safely.

        Not always useful. But in those cases it is a no op if its already contiguous 
        
        '''

    for chunk_id in range(num_chunks):
        start_idx = chunk_id * chunk_size # first chunk: start_idx = 0
        end_idx = min((chunk_id + 1) * chunk_size, BT) # first chunk: end_indx = 128
        _input_chunk = _input[start_idx:end_idx]  # chunk_size x H. # 128 x 768

        # when doing matmul, use the original precision (unlike how previously for loss accumulation we used fp32). So if values are in fp16, keep them in fp16
        logits_chunk = _input_chunk @ weight.t()  # chunk_size x V.  
        '''

        Deep dive into @
        Nothing to deep dive. its normal matmul :p. Need to transpose for the shapes we have.
        '''

        '''
        Deep dive into keeping original precision 

        On NVIDIA GPUs, PyTorch routes half/bfloat16 GEMMs to cuBLAS/cutlass/Tensor Cores that accumulate in FP32 even when inputs/outputs are fp16/bf16.
        So although your tensors are half/bfloat16:

        multiply → fp16/bf16

        accumulate (the sum over H) → fp32

        result is then cast back to fp16/bf16 for the output tensor.

        This fp32 accumulation is the key reason it’s numerically stable enough for training.
                '''
        if bias is not None: # for our example it is none
            logits_chunk = logits_chunk + bias

        target_chunk = target[start_idx:end_idx]  # chunk_size, (including padded ones if any)

        n_rows = logits_chunk.shape[0] # chunk_size -> 128

        # unreduced loss
        loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size, -> 128,
        z_loss_1d_slice = z_loss_1d[start_idx:end_idx] if return_z_loss else None

        # ensure _input and target are contiguous
        logits_chunk = logits_chunk.contiguous() # defensive as it is not a view
        target_chunk = target_chunk.contiguous()

        # Here we calculate the gradient of logits_chunk in place so we can save memory.
        liger_cross_entropy_kernel[(n_rows,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),  # always 1
            weight_ptr=ce_weight,
            loss_ptr=loss_1d_slice,
            z_loss_ptr=z_loss_1d_slice,
            loss_stride=loss_1d_slice.stride(-1),  # always 1
            n_cols=V,
            n_non_ignore=total_n_non_ignore,
            sum_non_ignore_weight=total_sum_non_ignore_ce_weight,
            weight_sum=ce_weight_sum,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            RETURN_Z_LOSS=return_z_loss,
            HAS_WEIGHT=True if ce_weight is not None else False,
            HAS_SOFTCAPPING=True if softcap is not None else False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        loss_1d[start_idx:end_idx] = loss_1d_slice
        if return_z_loss:
            z_loss_1d[start_idx:end_idx] = z_loss_1d_slice
        grad_logits_chunk = logits_chunk  # chunk_size x V

        grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

        if grad_weight is not None:
            torch.addmm(
                input=grad_weight,
                mat1=logits_chunk.t().to(
                    _input_chunk.dtype
                ),  # In an autocast scenario without bias, differing logits_chunk data types will cause an addmm operation error.
                mat2=_input_chunk,
                out=grad_weight,
                alpha=1.0,
                beta=1.0,
            )

        if bias is not None:
            torch.add(
                input=grad_bias,
                other=logits_chunk.sum(dim=0),
                out=grad_bias,
                alpha=1.0,
            )

    if reduction == "none": # example 1 : mean , ignore
        loss = loss_1d
        z_loss = z_loss_1d if return_z_loss else None

    else:
        loss = torch.sum(loss_1d)
        z_loss = torch.sum(z_loss_1d) if return_z_loss else None
    return loss, z_loss, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        # handle grad_weight
        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )

        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_bias,
                grad_bias.stride(-1),
                grad_output,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )
    return grad_input, grad_weight, grad_bias


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):

    '''
    Example 1: 
       
        _input: hidden_states right after the model forward: [batch, seq_len, d_model]; after reshape(-1, d_model) it becomes [(batch * seq_len), d_model]
        weight: model.lm_head.weight: [vocab_size, d_model] (PyTorch stores linear weights as [out_features, in_features])
        target: targets from target_ids.reshape(-1): length batch * seq_len.

    '''

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss: bool = False,
    ):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the _input and target
        for the backward pass.

        _input: (B*T, H) where B is batch size, T is sequence length, H is hidden dimension.
        target: (B*T) where each value is in [0, V-1]
        weight: (V, H) where V is the number of classes
        bias: (V) where V is the number of classes
        ce_weight: a manual rescaling weight given to each class. If given, has to be a Tensor of size V and floating point dtype
        ignore_index: the index to ignore in the target
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction: reduction to apply
        """

        loss, z_loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            ce_weight=ce_weight,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            return_z_loss=return_z_loss,
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if bias is not None else None,
        )
        ctx.return_z_loss = return_z_loss
        return loss, z_loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2):
        if ctx.return_z_loss:
            del grad_output2  # z_loss is only for logging
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (
            grad_input,
            grad_weight,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
