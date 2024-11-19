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


class ContextEncoder(torch.nn.Module):
    def __init__(self, context_dim, input_size=2):
        super(ContextEncoder, self).__init__()
        self.context_dim = context_dim

        self.hist_gru = torch.nn.GRU(input_size=6, hidden_size=context_dim, num_layers=1, batch_first=True)
        self.plan_gru = torch.nn.GRU(input_size=6, hidden_size=context_dim, num_layers=1, batch_first=True)

    def forward(self, hist, cond):
        _, hidden_hist = self.hist_gru(hist)
        _, hidden_cond = self.plan_gru(cond)

        condition = torch.cat([hidden_hist[0], hidden_cond[0]], dim=-1)

        return condition


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out, bias=False)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret


class TransformerConcatLinear(nn.Module):
    def __init__(self, context_dim, T):
        super().__init__()
        self.context = ContextEncoder(context_dim//2)
        self.time_embedding = TimeEmbedding(T, context_dim // 4, context_dim)

        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=100)
        self.concat1 = ConcatSquashLinear(4, 2*context_dim, context_dim*2)

        self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=4)

        self.concat3 = ConcatSquashLinear(2*context_dim, context_dim, context_dim*2)
        self.concat4 = ConcatSquashLinear(context_dim, context_dim//2, context_dim*2)
        self.linear = ConcatSquashLinear(context_dim//2, 4, context_dim*2)

    def forward(self, t, x, v, context, mask):
        batch_size = x.size(0)
        context = context.view(batch_size, 1, -1)

        time_emb = self.time_embedding(t).unsqueeze(1)
        ctx_emb = torch.cat([time_emb, context], dim=-1)
        x = self.concat1(ctx_emb, x)
        final_emb = x.permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb).permute(1, 0, 2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)

        return self.linear(ctx_emb, trans)
