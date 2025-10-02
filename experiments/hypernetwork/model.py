import torch
import torch.nn as nn
import torch.nn.functional as F


def split_heads(x, n_heads):
    B, T, H = x.shape
    d = H // n_heads
    return x.view(B, T, n_heads, d).transpose(1, 2)  # [B, nH, T, d]

def merge_heads(x):
    B, nH, T, d = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, nH * d)  # [B, T, H]


class BiasedMHA(nn.Module):
    """
    Multihead attention with additive 'bias' on attention scores.
    bias: [n_heads, T_q, T_k] or [B, n_heads, T_q, T_k] (float, added to logits).
    """
    def __init__(self, hid, n_heads, dropout=0.1):
        super().__init__()
        assert hid % n_heads == 0
        self.hid, self.n_heads, self.d = hid, n_heads, hid // n_heads
        self.q_proj = nn.Linear(hid, hid)
        self.k_proj = nn.Linear(hid, hid)
        self.v_proj = nn.Linear(hid, hid)
        self.o_proj = nn.Linear(hid, hid)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, bias=None):
        # q,k,v: [B, T, hid]
        B, Tq, _ = q.shape
        Tk = k.size(1)

        qh = split_heads(self.q_proj(q), self.n_heads)  # [B,nH,Tq,d]
        kh = split_heads(self.k_proj(k), self.n_heads)  # [B,nH,Tk,d]
        vh = split_heads(self.v_proj(v), self.n_heads)  # [B,nH,Tk,d]

        # scaled dot-prod attention with additive bias
        # attn_mask in PyTorch 2.x can be float and broadcastable; we provide [B*nH,Tq,Tk]
        attn_mask = None
        if bias is not None:
            if bias.dim() == 3:            # [nH, Tq, Tk]
                bias = bias.unsqueeze(0).expand(B, -1, -1, -1)
            # reshape for SDPA: [B*nH, Tq, Tk]
            attn_mask = bias.reshape(B * self.n_heads, Tq, Tk)

        # Flatten heads for SDPA
        qf = qh.reshape(B * self.n_heads, Tq, self.d)
        kf = kh.reshape(B * self.n_heads, Tk, self.d)
        vf = vh.reshape(B * self.n_heads, Tk, self.d)

        out = F.scaled_dot_product_attention(
            qf, kf, vf,
            attn_mask=attn_mask,   # added to logits before softmax
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=False
        )  # [B*nH, Tq, d]

        out = out.reshape(B, self.n_heads, Tq, self.d)
        out = merge_heads(out)              # [B, Tq, hid]
        return self.o_proj(out)


class RelationBias(nn.Module):
    """
    Learns per-head scalars for simple 2D relations on (dim, level).
    Produces a bias tensor added to attention scores in self-attn.
    """
    def __init__(self, n_heads):
        super().__init__()
        # learned weights per head for each relation
        self.w_same_dim  = nn.Parameter(torch.zeros(n_heads))   # encourages cores within same dimension
        self.w_same_lvl  = nn.Parameter(torch.zeros(n_heads))   # encourages cores at same level
        self.w_adj_lvl   = nn.Parameter(torch.zeros(n_heads))   # encourages |lvl_i - lvl_j| == 1
        # # optional: distance decay on level differences (smooth bias)
        self.w_lvl_dist  = nn.Parameter(torch.zeros(n_heads))   # multiplies a normalized distance kernel

    def forward(self, dim_idx, lvl_idx):
        """
        dim_idx, lvl_idx: [K] int tensors on the right device
        returns bias: [n_heads, K, K]
        """
        K = dim_idx.numel()
        di = dim_idx.view(1, K)
        li = lvl_idx.view(1, K)

        same_dim = (di.T == di).float()         # [K,K]
        same_lvl = (li.T == li).float()         # [K,K]
        adj_lvl  = (li.T - li).abs().eq(1).float()

        # level distance kernel in [0,1], 1 on diag, decays with |Δ|
        d = (li.T - li).abs().float()
        # normalize to [0,1] by L-1 when L>1; if L==1, kernel is ones
        Lm1 = max(int(lvl_idx.max().item()), 0)
        lvl_kernel = torch.ones_like(d) if Lm1 == 0 else 1.0 - (d / (Lm1 + 1e-6))

        # stack per relation, then weight per head
        # shape to [1,K,K] then broadcast to [nH,K,K]
        def w_expand(w): return w.view(-1, 1, 1)
        bias = (
                w_expand(self.w_same_dim) * same_dim.unsqueeze(0) +
                w_expand(self.w_same_lvl) * same_lvl.unsqueeze(0) +
                w_expand(self.w_adj_lvl)  * adj_lvl.unsqueeze(0) +
                w_expand(self.w_lvl_dist) * lvl_kernel.unsqueeze(0)
        )
        return bias  # [nH, K, K]


class FeedForward(nn.Module):
    def __init__(self, hid, mult=4, dropout=0.1):
        super().__init__()
        inner = max(512, mult * hid)
        self.net = nn.Sequential(
            nn.Linear(hid, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hid),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class QTTSeqCondAttn(nn.Module):
    """
    Transformer encoder with alternating:
      [relation-biased self-attn on data tokens] + [cross-attn to cond tokens] + FFN.

    Outputs alpha: [B, K, M] and hidden h: [B, K, hid].
    """
    def __init__(
            self,
            d, L, cond_dim, emb=128, hid=256, M=8,
            num_layers=4, n_heads=8, dropout=0.1,
            cond_tokens=4
    ):
        super().__init__()
        self.d, self.L, self.K = d, L, d * L
        self.M = M
        self.hid = hid
        self.n_heads = n_heads
        self.cond_tokens = cond_tokens

        # tokenization
        self.dim_emb = nn.Embedding(d, emb)
        self.lvl_emb = nn.Embedding(L, emb)
        self.in_proj = nn.Linear(emb, hid)

        # cond vector -> P tokens
        assert cond_dim >= 0
        P = max(cond_tokens, 1) if cond_dim > 0 else 0
        self.P = P
        if P > 0:
            self.cond_proj = nn.Linear(cond_dim, P * hid)

        # blocks
        self.self_attn = nn.ModuleList([BiasedMHA(hid, n_heads, dropout) for _ in range(num_layers)])
        self.cross_attn = nn.ModuleList([BiasedMHA(hid, n_heads, dropout) for _ in range(num_layers)])
        self.ff = nn.ModuleList([FeedForward(hid, mult=4, dropout=dropout) for _ in range(num_layers)])

        self.ln1 = nn.ModuleList([nn.LayerNorm(hid) for _ in range(num_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(hid) for _ in range(num_layers)])
        self.ln3 = nn.ModuleList([nn.LayerNorm(hid) for _ in range(num_layers)])

        self.rel_bias = RelationBias(n_heads)

        # head
        self.head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, M),
        )

        # precompute (j, l) -> buffers moved to device at forward
        tokens = [(j, l) for j in range(d) for l in range(L)]
        self.register_buffer("dim_idx_cpu", torch.tensor([t[0] for t in tokens], dtype=torch.long), persistent=False)
        self.register_buffer("lvl_idx_cpu", torch.tensor([t[1] for t in tokens], dtype=torch.long), persistent=False)

    def forward(self, cond):
        B = cond.size(0) if isinstance(cond, torch.Tensor) and cond.numel() > 0 else 1
        device = cond.device if isinstance(cond, torch.Tensor) else next(self.parameters()).device

        # data tokens
        dim_idx = self.dim_idx_cpu.to(device)  # [K]
        lvl_idx = self.lvl_idx_cpu.to(device)  # [K]
        x = self.dim_emb(dim_idx) + self.lvl_emb(lvl_idx)   # [K, emb]
        x = self.in_proj(x).unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, K, hid]

        # cond tokens
        if self.P > 0:
            ptoks = self.cond_proj(cond).view(B, self.P, self.hid)       # [B, P, hid]
        else:
            ptoks = None

        # precompute relation bias once per forward for self-attn
        attn_bias = self.rel_bias(dim_idx, lvl_idx)  # [nH, K, K]

        # stack blocks
        h = x
        for i in range(len(self.self_attn)):
            # 1) relation-biased self-attn over data tokens
            h = h + self.self_attn[i](
                self.ln1[i](h), self.ln1[i](h), self.ln1[i](h),
                bias=attn_bias  # added to logits per head
            )
            # 2) cross-attn: queries = data tokens, keys/values = cond tokens
            if ptoks is not None:
                h = h + self.cross_attn[i](self.ln2[i](h), self.ln2[i](ptoks), self.ln2[i](ptoks), bias=None)
            # 3) feedforward
            h = h + self.ff[i](self.ln3[i](h))

        # per-position mixture
        logits = self.head(h)         # [B, K, M]
        alpha = F.softmax(logits, dim=-1)
        return alpha, h


class QTTCoreBank(nn.Module):
    def __init__(self, r=32, M=8, orth_penalty=1e-4):
        super().__init__()
        # basis cores: [M, 2, r, r]
        self.B = nn.Parameter(torch.randn(M, 2, r, r) / (r**0.5))
        self.orth_penalty = orth_penalty

    def make_core(self, alpha):  # alpha: [B, M] -> returns tuple ([B, r, r], [B, r, r])
        # weighted sum over basis per batch
        # B: [M, 2, r, r], alpha: [B, M]
        # out for bit=0: [B, r, r]
        g0 = torch.einsum('bm,mij->bij', alpha, self.B[:,0])
        g1 = torch.einsum('bm,mij->bij', alpha, self.B[:,1])
        return g0, g1

    def orth_loss(self):
        if self.orth_penalty <= 0:
            return torch.tensor(0.0, device=self.B.device)
        loss = 0.0
        for b in (0, 1):
            # [M, r, r]
            Bslice = self.B[:, b]
            BtB = torch.matmul(Bslice.transpose(-1, -2), Bslice)  # [M, r, r]
            I = torch.eye(BtB.size(-1), device=BtB.device).expand_as(BtB)
            loss = loss + (BtB - I).pow(2).mean()
        return self.orth_penalty * loss


class QTTSeq(nn.Module):
    """
    Sequence model that outputs per-core mixture weights alpha_k
    conditioned on 'cond' of shape [B, conditional_dim].
    """
    def __init__(self, d, L, conditional_dim, embedding_dim=128, hidden_dim=256, M=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.d, self.L, self.K = d, L, d * L
        self.conditional_dim = conditional_dim
        self.M = M

        self.dimension_embedding = nn.Embedding(d, embedding_dim)
        self.bit_level_embedding = nn.Embedding(L, embedding_dim)
        self.conditional_proj = nn.Linear(conditional_dim, hidden_dim)
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, M)
        )

        # precompute token indices once
        # interleaved by dimension then level, can be the other way round depending on induction bias preference (whatever that means)
        tokens = []
        for j in range(d):
            for l in range(L):
                tokens.append((j, l))
        self.register_buffer(
            "dim_idx_cpu", torch.tensor([t[0] for t in tokens], dtype=torch.long),
            persistent=False
        )
        self.register_buffer(
            "lvl_idx_cpu", torch.tensor([t[1] for t in tokens], dtype=torch.long),
            persistent=False
        )

    def forward(self, cond):
        """
        cond: [B, conditional_dim]
        returns:
          alpha: [B, K, M] mixture per position
          h:     [B, K, hidden_dim] hidden states (for optional readouts)
        """
        B = cond.size(0)
        device = cond.device

        dim_idx = self.dim_idx_cpu.to(device)
        lvl_idx = self.lvl_idx_cpu.to(device)

        # token embeddings, takes the dim and level indices, converts up to hid
        tok = self.dimension_embedding(dim_idx) + self.bit_level_embedding(lvl_idx)  # [K, embedding_dim]
        tok = self.embedding_proj(tok)  # [K, hidden_dim]
        tok = tok.unsqueeze(0).repeat(B, 1, 1)  # [B, K, hid]

        # condition projection up to hidden_dim (same as tok dim) in order to combine
        conditional_bias = self.conditional_proj(cond).unsqueeze(1)  # [B, 1, hidden_dim]
        x = tok + conditional_bias  # broadcast to all K so that [B, K, hidden_dim]

        # run through GRU
        h, _ = self.gru(x)  # [B, K, hidden_dim]

        # Run through MLP head to get Mixture weights
        logits = self.head(h)  # [B, K, M]
        alpha = F.softmax(logits, dim=-1)
        return alpha, h


class QTTGenerator(nn.Module):
    """
    produces dense outputs atm, need to update to do sampled contraction loss apparently
    """
    def __init__(self, d=5, N=128, r=32, M=8, cond_dim=0, orth_penalty=1e-4):
        super().__init__()
        assert (N & (N - 1)) == 0, "N must be a power of 2"
        L = N.bit_length() - 1
        self.d, self.N, self.L, self.K = d, N, L, d * L
        self.bank = QTTCoreBank(r=r, M=M, orth_penalty=orth_penalty)
        # self.seq = QTTSeq(d, L, conditional_dim=cond_dim, embedding_dim=128, hidden_dim=256, M=M, num_layers=2,
        #                   dropout=0.1)

        self.seq = QTTSeqCondAttn(
            d, L, cond_dim=cond_dim,
            emb=128, hid=256, M=M,
            num_layers=4, n_heads=8, dropout=0.1,
            cond_tokens=4,    # 2–8 is typical
        )

    def forward_sampled(self, cond, bits):
        """
        cond: [B, conditional_dim]
        bits: [B, S, K] with 0/1 entries per sampled index
        returns: [B, S] predicted scalar values
        """
        B, S, K = bits.shape
        assert K == self.K, "bits last dim must equal K=d*log2(N)"
        device = cond.device
        bits = bits.to(device)

        alpha, _ = self.seq(cond)  # [B, K, M]
        M = alpha.size(-1)
        ak = alpha.reshape(B * K, M)
        g0, g1 = self.bank.make_core(ak)  # each [B*K, r, r]
        r = g0.size(-1)
        g0 = g0.view(B, K, r, r)
        g1 = g1.view(B, K, r, r)

        if not hasattr(self, "u"):
            self.u = torch.nn.Parameter(torch.ones(r, device=device) / r**0.5)
            self.v = torch.nn.Parameter(torch.ones(r, device=device) / r**0.5)

        # contraction
        L = self.u.view(1, 1, 1, r).expand(B, S, 1, r).contiguous()  # [B, S, 1, r]
        for k in range(K):
            # Select G_k^{(b)} for each (B,S)
            # g_sel: [B, S, r, r]
            gk0 = g0[:, k]  # [B, r, r]
            gk1 = g1[:, k]  # [B, r, r]
            bk = bits[:, :, k]  # [B, S]
            # Gather: stack [B,2,r,r] then take index per (B,S)
            Gk = torch.stack([gk0, gk1], dim=1)  # [B, 2, r, r]
            # Convert bk to indices: [B,S] -> gather across dim=1
            # Build [B,S,r,r] by advanced indexing
            Gk_sel = Gk.gather(1, bk.unsqueeze(-1).unsqueeze(-1).expand(B, S, r, r))
            # Multiply: [B,S,1,r] x [B,S,r,r] -> [B,S,1,r]
            L = torch.matmul(L, Gk_sel)

        # Finish with right boundary
        out = torch.matmul(L, self.v.view(1, 1, r, 1))  # [B,S,1,1]
        return out.view(B, S)

    def orth_loss(self):
        return self.bank.orth_loss()
