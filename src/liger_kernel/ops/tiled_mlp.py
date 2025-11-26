import math

from typing import Callable
from typing import List
from typing import Optional

import torch

from liger_kernel.ops.utils import ensure_contiguous


class LigerTiledMLPFunction(torch.autograd.Function):
    """
    Based on DeepSpeed's TiledMLP:
    https://github.com/deepspeedai/DeepSpeed/blob/v0.18.2/deepspeed/runtime/sequence_parallel/ulysses_sp.py#L838

    Perform a tiled MLP computation to massively reduce memory usage needed to compute MLP
    when using very long sequence lengths.

    This module re-computes `forward` in the `backward`. So the `forward` occurs twice each iteration.
    And if you're using activation checkpointing it then occurs thrice.

    Args:
        fn: the function to call on sharded inputs (e.g., mlp.forward)
        mlp_module: the MLP nn.Module object
        x: the input to MLP.forward (hidden_states)
        shards: how many shards to use
        compute_params: a list of weights engaged in the compute

    Returns:
        the computed hidden_states
    """

    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        fn: Callable,
        mlp_module: torch.nn.Module,
        x: torch.Tensor,
        shards: int,
        compute_params: Optional[List[torch.nn.Parameter]] = None,
    ) -> torch.Tensor:
        ctx.fn = fn
        ctx.mlp_module = mlp_module
        ctx.shards = shards
        ctx.save_for_backward(x)

        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size] (moe experts)
        x_shards = list(torch.chunk(x, chunks=shards, dim=-2))
        '''
        Forward deliberately ran each shard under torch.no_grad(), so its activations/graph were never kept. By backward time there’s nothing to backprop through unless we rebuild the graph. Recomputing output = fn(mlp_module, x_shard) inside torch.enable_grad() regenerates the shard’s activations and graph on-the-fly; then autograd.backward uses that graph with the shard’s upstream grad to fill the preallocated x_grad slice. This trades extra compute (~2× MLP) for much lower activation memory, which is the whole point of the tiled MLP.

        Activation footprint: baseline vs tiled

Baseline (no tiling, no recompute): saved tensors are roughly x (hidden), plus a/b from gate+up (intermediate each), plus the down-proj input c (intermediate). Elements per token ≈ hidden_size + 3*intermediate_size. In bf16/fp16 that’s ×2 bytes.
Tiled with recompute: forward saves only x (hidden). During backward each shard temporarily materializes a/b/c for that shard only, then frees them.
Example (bf16, B=1, seq=8k, hidden=4096, intermediate=11008 ~2.7× hidden):

Baseline saved activations: (4096 + 3*11008) * 8192 ≈ 304M elements ≈ 608 MB.
Tiled saved activations: 4096 * 8192 ≈ 33.6M elements ≈ 67 MB.
Per-shard working set during backward (default shards=ceil(seq/hidden)=2 → shard_len≈4096): 3*11008*4096 ≈ 135M elements ≈ 270 MB live one shard at a time. Peak ≈ 67 MB (saved) + 270 MB (working) ≈ 337 MB, still ~1.8× smaller than baseline; if num_shards is larger, the working set drops proportionally.
Longer seq (seq=32k, same dims):

Baseline saved: ≈2.4 GB.
Tiled saved: ≈268 MB; working set per shard still ≈270 MB. Peak ≈538 MB → ~4–5× smaller.
So tiling removes the need to keep full-sequence intermediate activations; only the input is saved, and recompute + sharding bounds the transient activations to one shard at a time.

        '''
        with torch.no_grad(): # run deliberately in forward so grad is not kept
            output_shards = [fn(mlp_module, x_shard) for x_shard in x_shards]
        output_unsharded = torch.cat(output_shards, dim=-2)

        return output_unsharded

    @staticmethod
    @ensure_contiguous
    def backward(ctx, *grads) -> tuple:
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        mlp_module = ctx.mlp_module
        shards = ctx.shards

        x_requires_grad = x.requires_grad
        '''
        x.detach() drops any existing autograd history on the saved input so it becomes a fresh leaf for the recompute. That way:

The backward recomputation builds a new graph from x_shard to output without trying to reuse whatever graph x originally came from.
We can manually wire .grad to x_grad and return the gradient upstream, instead of autograd attempting to backprop through an old graph.
After detaching, they restore requires_grad_ to the original value so grads still flow to x (via the returned x_grad) but not through any prior history.
        '''
        x = x.detach()
        # detach() unsets x.requires_grad, so restore it
        x.requires_grad_(x_requires_grad)

        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size] (moe experts)
        hidden_size = x.shape[-1]
        x_shape_orig = x.shape

        # flatten bs+seqlen to avoid having stride issues when narrowing into seqlen w/ bs>1
        x = x.view(-1, hidden_size)
        incoming_grad = grads[0].view(-1, hidden_size)
        x_grad = torch.zeros_like(x)

        x_shards = list(torch.chunk(x, chunks=shards, dim=0))

        for i, x_shard in enumerate(x_shards):
            x_shard.requires_grad_(x_requires_grad)

            # if seqlen is not exactly divisible by shards the last step will be shorter than shard_step
            shard_step = x_shards[i].shape[0]
            shard_offset = i * x_shards[0].shape[0]

            x_shard.grad = x_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)
            incoming_grad_shard = incoming_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)

            with torch.enable_grad():
                output = fn(mlp_module, x_shard)
            '''
            torch.autograd.backward(output, incoming_grad_shard) runs backprop for just this shard: it treats incoming_grad_shard as the upstream gradient for output and walks the recomputed graph to produce grads w.r.t. the shard’s inputs. Because x_shard.grad was wired to a slice of x_grad, the resulting input grads land directly in the right spot of the global buffer.
            '''
            torch.autograd.backward(output, incoming_grad_shard)

        # unflatten
        x_grad = x_grad.view(x_shape_orig)

        # None for non differentiable inputs in the forward
        return (None, None, x_grad, None, None)


def apply_tiled_mlp(
    fn: Callable,
    mlp_module: torch.nn.Module,
    x: torch.Tensor,
    num_shards: Optional[int] = None,
    compute_params: Optional[List[torch.nn.Parameter]] = None,
) -> torch.Tensor:
    """
    Apply tiled MLP computation for memory efficiency.

    Args:
        fn: the function to call on sharded inputs (e.g., lambda module, x: module(x))
        mlp_module: the MLP nn.Module object
        x: the input tensor with shape [bs, seqlen, hidden_size] or [seqlen, hidden_size]
        num_shards: number of shards to use. If None, automatically calculated as ceil(seqlen / hidden_size)
        compute_params: list of parameters for DeepSpeed ZeRO optimization

    Returns:
        output tensor with the same shape as input
    """
    if num_shards is None:
        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size]
        hidden_size = x.shape[-1]
        seqlen = x.shape[-2]
        num_shards = math.ceil(seqlen / hidden_size)

    # Ensure num_shards is at least 1
    num_shards = max(1, num_shards)

    return LigerTiledMLPFunction.apply(
        fn,
        mlp_module,
        x,
        num_shards,
        compute_params,
    )
