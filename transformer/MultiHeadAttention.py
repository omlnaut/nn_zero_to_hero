import torch
import torch.nn as nn

from .SelfAttentionHead import SelfAttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, token_size, block_size):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(head_size, token_size, block_size)
                for _ in range(n_heads)
            ]
        )

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)

        return concat
