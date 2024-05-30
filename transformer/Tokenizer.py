import torch


class Tokenizer:
    def __init__(self, text: str):
        self.text = text
        self.tokens = sorted(list(set(text)))

        self.stoi = {ch: i for i, ch in enumerate(self.tokens)}
        self.itos = {i: ch for i, ch in enumerate(self.tokens)}

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def decode(self, tensor: torch.Tensor) -> str:
        return "".join([self.itos[int(i.item())] for i in tensor])

    @property
    def vocab_size(self):
        return len(self.tokens)
