import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class NoisyReward(torch.nn.Module):
    def __init__(self, context_dim, T):
        super(NoisyReward, self).__init__()

        self.context_dim = context_dim
        self.time_embed = TimeEmbedding(T, context_dim // 4, context_dim)
        self.state_embed = nn.Sequential(nn.Linear(16, context_dim),
                                         nn.ReLU(),
                                         nn.Linear(context_dim, context_dim))
        self.out_mlp = nn.Sequential(nn.Linear(context_dim*2, context_dim),
                                     nn.ReLU(),
                                     nn.Linear(context_dim, context_dim),
                                     nn.ReLU(),
                                     nn.Linear(context_dim, 1))

    def forward(self, x, t):
        time_emb = self.time_embed(t).unsqueeze(1).repeat(1, x.shape[1], 1)
        x = self.state_embed(x)
        x = torch.cat([x, time_emb], dim=-1)

        x = self.out_mlp(x)

        return x

