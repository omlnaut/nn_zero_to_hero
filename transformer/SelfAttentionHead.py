import torch


class SelfAttentionHead(torch.nn.Module):
    def __init__(self, head_size, token_emb_size, block_size):
        super().__init__()
        self.key = torch.nn.Linear(token_emb_size, head_size)
        self.query = torch.nn.Linear(token_emb_size, head_size)
        self.value = torch.nn.Linear(token_emb_size, head_size)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = k @ q.transpose(-2, -1) * C ** (-0.5)
        wei.masked_fill_(self.tril[:T, :T] == 0, float("-inf"))
        wei = torch.nn.functional.softmax(wei, dim=-1)

        return wei @ v
