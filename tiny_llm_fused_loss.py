"""
Minimal causal language model example that uses the Liger fused linear cross
entropy loss to combine the final projection and token-level cross entropy
computation into a single Triton kernel.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


class TinyCausalLM(nn.Module):
    """
    Lightweight decoder-only Transformer that outputs hidden states for every
    token position. The final language-modeling projection is left outside of
    the module so the fused loss can materialize logits chunk by chunk.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    activation="gelu",
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:  # Enforce positional embedding table limit.
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )

        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :seq_len]

        for layer in self.layers:
            x = layer(x, src_mask=attn_mask)

        x = self.norm(x)
        return x


def main() -> None:
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit(
            "This example requires a CUDA-capable device so Triton kernels can run."
        )

    vocab_size = 2048
    seq_len = 64
    batch_size = 8

    model = TinyCausalLM(vocab_size=vocab_size, max_seq_len=seq_len).to(device)
    # The fused loss projects hidden states with lm_head weights and executes cross-entropy in one Triton kernel.
    loss_fn = LigerFusedLinearCrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

    def sample_batch() -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Generates a toy batch of token IDs and next-token targets. Targets are
        shifted by one position so the loss is computed on predicting the next
        token, just like causal language modeling.
        """

        data = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_len + 1),
            device=device,
            dtype=torch.long,
        )
        inputs = data[:, :-1]
        targets = data[:, 1:]
        return inputs, targets

    num_steps = 20
    for step in range(1, num_steps + 1):
        input_ids, target_ids = sample_batch()

        hidden_states = model(input_ids)  # shape: [B, T, H]
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        targets = target_ids.reshape(-1)

        # We intentionally avoid materializing logits or softmax; the fused kernel performs that work internally.
        '''
        Shapes: 
        model.lm_head.weight: [vocab_size, d_model] (PyTorch stores linear weights as [out_features, in_features])
       
        hidden_states right after the model forward but before the : [batch, seq_len, d_model]; after reshape(-1, d_model) it becomes [(batch * seq_len), d_model]

        targets from target_ids.reshape(-1): length batch * seq_len.

        '''
        loss = loss_fn(model.lm_head.weight, hidden_states, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 5 == 0:
            perplexity = math.exp(loss.item())
            head_grad_norm = model.lm_head.weight.grad.norm().item()
            print(
                f"step={step:02d} loss={loss.item():.4f} ppl={perplexity:.2f} "
                f"lm_head_grad_norm={head_grad_norm:.4f}"
            )

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
