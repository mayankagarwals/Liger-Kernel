import torch

from liger_kernel.ops.cross_entropy import cross_entropy_forward


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to launch the Triton cross entropy kernel.")

    batch, seq_len, vocab = 2, 3, 5
    bt = batch * seq_len
    logits = torch.randn(bt, vocab, device=device, dtype=torch.float32)
    targets = torch.tensor([1, 3, 4, -100, 0, 2], device=device, dtype=torch.int64)

    loss, z_loss, grads = cross_entropy_forward(
        logits,
        targets,
        weight=None,
        ignore_index=-100,
        lse_square_scale=0.2,
        label_smoothing=0.1,
        reduction="mean",
        softcap=None,
        return_z_loss=True,
    )

    torch.cuda.synchronize()

    print("loss:", loss.item())
    print("z_loss:", z_loss.item() if z_loss is not None else None)
    print("grad sample:", grads[0, :5].detach().cpu())


if __name__ == "__main__":
    main()
