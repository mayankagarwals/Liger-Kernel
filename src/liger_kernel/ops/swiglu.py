import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings
from liger_kernel.ops.utils import ensure_contiguous


@triton.jit
def silu(x):
    '''
    why did we not have to pass block size etc in silu(x)
silu in src/liger_kernel/ops/swiglu.py isn’t a separate kernel launch; it’s a tiny, inlined device function. The kernel _swiglu_forward_kernel already fixed BLOCK_SIZE and num_warps when it built a_row/b_row (vectors of length BLOCK_SIZE). Calling silu(a_row) just applies x * tl.sigmoid(x) to that vector inside the same program—no new grid or launch configuration needed. Block size/warps are only required at the kernel boundary, not for an elementwise helper used inside.
how is such a boundary differentiated?
The boundary is the kernel launch. Anything annotated @triton.jit that you invoke with the launch syntax _kernel[(grid,)](..., BLOCK_SIZE=..., num_warps=...) is a kernel entry point—host code sets grid/warps/block size there. Functions called inside that kernel (like silu(a_row)) are just inlined device code; they run with the same threads/program_id and don’t accept launch parameters. So block size/warps are specified only at the kernel entry point, not for inner helpers.
    '''
    return x * tl.sigmoid(x) # swish! tl.sigmoid is elementwise so is multiplication


@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    '''
    The gate matmul typically runs in fp16/bf16. Triton’s tl.sigmoid is implemented in fp32 (and more stable there), so they promote a_row to float32 before calling it. After computing SiLU, they cast that activation back to b_row’s dtype so the final product matches the original matmul/output dtype (fp16/bf16) instead of staying in fp32, which keeps the output consistent and avoids extra bandwidth/compute from higher precision.
    Triton’s guidance is mostly implicit: transcendental functions (tl.exp, tl.sigmoid, tl.log, etc.) promote to fp32 internally for accuracy/stability, and the tutorials/best‑practices note that reductions and nonlinearities should accumulate/compute in fp32 even when inputs are fp16/bf16. There isn’t a per‑op speed table, but the “Programmer’s Guide” and tutorial examples (e.g., softmax) show the common pattern: cast to fp32 for the math, then cast back to the desired dtype for storage.
    Yes, it’s meant to be faster than the default PyTorch path. PyTorch does SiLU and the subsequent multiply as separate pointwise kernels, writing the gate activation to memory and then reading it back to multiply with up. Liger’s LigerSiLUMulFunction (src/liger_kernel/ops/swiglu.py) fuses those two steps into one Triton kernel: it loads gate and up once, computes silu(gate) in fp32 for stability, multiplies by up, and stores the result—one launch, one write, less bandwidth.

    '''
    c_row = silu(a_row).cast(b_row.dtype) * b_row. # elementwise b*swish(a)
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(dc_ptr, a_ptr, b_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    # recomputation to save memory
    sig_a = tl.sigmoid(a_row)
    silu_a = a_row * sig_a # c = b. silu(a). dc/db = silu(a) = a * sig_a 
    db_row = dc_row * silu_a # dL/db = dL/dc * dc/db = dc_row * silu_a
    da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row # refer notion for derivation 

    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)


def swiglu_forward(a, b):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape) # only c is returned to the caller


def swiglu_backward(a, b, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a.view(*ori_shape), b.view(*ori_shape)


class LigerSiLUMulFunction(torch.autograd.Function):

    '''
    Tips when inheriting torch.autograd.Function
    This will be used in pytorch graph as MyFunc.apply(...)

    graph -> inputs -> forward -> output
              grad of input w.r.t loss    <- backward <- grad of output w.r.t loss

    backward takes one grad per forward output (including Nones) and returns one grad per forward input; use None for non-differentiable inputs

    For example in the below case 
    forward input : a,b. So backward ouptut should be two values 
    forward ouptut : c . So backward input should take a grad (grad of c w.r.t loss)

    Save carefully: use ctx.save_for_backward for tensors; stash small metadata on ctx (shapes, flags). Don’t save huge tensors if you can recompute cheaper—trade memory vs compute deliberately.

    '''

    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, c = swiglu_forward(a, b)
        '''
        ctx is the per-call autograd context PyTorch passes into your torch.autograd.Function.forward. It exists only for that invocation. You use it to stash tensors via ctx.save_for_backward(...) and can attach other metadata (ctx.foo = ...) for use in backward. Outside of that, it’s not used.

        '''
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        a, b = swiglu_backward(a, b, dc)
        return a, b
