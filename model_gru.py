import torch
import torch.nn as nn
import numpy as np

# Re-use these unchanged from model.py
from model import make_splits, ByteDataloader, _aligned_len

def BoaConstrictorGRU(d_model=256, num_layers=4, vocab_size=256, device="cuda"):
    """
    GRU-based backbone replacement for BoaConstrictor.
    Replaces MambaBlock with GRUBlock using standard PyTorch nn.GRU.
    No custom CUDA kernels required — runs on any PyTorch installation.
    """
    IS_CUDA = torch.cuda.is_available() and device == "cuda"
    device = "cuda" if IS_CUDA else "cpu"

    class GRUBlock(nn.Module):
        """
        Drop-in replacement for MambaBlock.
        Same interface: forward(x, inference_params=None) -> x
        Uses a single-layer GRU with residual connection and feedforward.
        """
        def __init__(self, d_model: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            # GRU: input and hidden size both d_model
            # batch_first=True so input shape is [B, L, D]
            self.gru = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True
            )
            self.ln2 = nn.LayerNorm(d_model)
            # Same feedforward as MambaBlock
            self.ff = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )

        def forward(self, x, inference_params=None):
            # x: [B, L, D]
            y = self.ln1(x)
            # GRU returns (output, hidden_state) — we only need output
            y, _ = self.gru(y)
            y = self.ln2(y)
            y = self.ff(y)
            # Residual connection — same as MambaBlock
            return x + y

        def init_cache(self, batch_size: int, device):
            # GRU hidden state: [num_layers, B, D]
            return torch.zeros(1, batch_size, 
                             self.gru.hidden_size, device=device)

        def step(self, x, hidden):
            # x: [B, D] — single token step for streaming
            y = self.ln1(x)
            # GRU expects [B, 1, D] for single step
            y, hidden = self.gru(y.unsqueeze(1), hidden)
            y = y.squeeze(1)  # back to [B, D]
            y = self.ln2(y)
            y = self.ff(y)
            return x + y, hidden

    class BoaByteGRUPredictor(nn.Module):
        """
        GRU-based byte predictor.
        Same interface as BoaBytePredictor but uses GRUBlock instead of MambaBlock.
        """
        def __init__(self, d_model=256, num_layers=4, vocab_size=256):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            # Replace MambaBlock with GRUBlock here
            self.blocks = nn.ModuleList([
                GRUBlock(d_model) for _ in range(num_layers)
            ])
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, vocab_size)
            )

        def forward(self, x, inference_params=None):
            # inference_params ignored — GRU doesn't use it
            # kept for interface compatibility with original
            h = self.embedding(x)  # [B, L, D]
            for blk in self.blocks:
                h = blk(h)
            return self.head(h)  # [B, L, 256]

        @torch.inference_mode()
        def init_stream(self, max_len: int, batch_size: int = 1, 
                       device=None, dtype=None):
            # Return list of per-block hidden states
            d = device or ("cuda" if IS_CUDA else "cpu")
            return [blk.init_cache(batch_size, d) for blk in self.blocks]

        @torch.inference_mode()
        def step(self, byte_t: torch.LongTensor, caches) -> torch.Tensor:
            # byte_t: [B] -> logits: [B, 256]
            h = self.embedding(byte_t)  # [B, D]
            for i, blk in enumerate(self.blocks):
                h, caches[i] = blk.step(h, caches[i])
            return self.head(h)  # [B, 256]

    model = BoaByteGRUPredictor(
        d_model=d_model, 
        num_layers=num_layers, 
        vocab_size=vocab_size
    )
    return model