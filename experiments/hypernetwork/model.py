import torch
import torch.nn as nn
import torch.nn.functional as F

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
    conditioned on 'cond' of shape [B, cond_dim].
    """
    def __init__(self, d, L, cond_dim, emb=128, hid=256, M=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.d, self.L, self.K = d, L, d * L
        self.cond_dim = cond_dim
        self.M = M

        self.dim_emb = nn.Embedding(d, emb)
        self.lvl_emb = nn.Embedding(L, emb)
        self.cond_proj = nn.Linear(cond_dim, hid)
        self.in_proj = nn.Linear(emb, hid)
        self.gru = nn.GRU(hid, hid, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, M)
        )

        # precompute token indices once (will move to device at forward)
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
        cond: [B, cond_dim]
        returns:
          alpha: [B, K, M] mixture per position
          h:     [B, K, hid] hidden states (for optional readouts)
        """
        B = cond.size(0)
        device = cond.device

        dim_idx = self.dim_idx_cpu.to(device)
        lvl_idx = self.lvl_idx_cpu.to(device)

        tok = self.dim_emb(dim_idx) + self.lvl_emb(lvl_idx)  # [K, emb]
        tok = self.in_proj(tok)                              # [K, hid]
        tok = tok.unsqueeze(0).repeat(B, 1, 1)               # [B, K, hid]

        cond_bias = self.cond_proj(cond).unsqueeze(1)        # [B, 1, hid]
        x = tok + cond_bias                                  # broadcast to all K

        h, _ = self.gru(x)                                   # [B, K, hid]
        logits = self.head(h)                                # [B, K, M]
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
        self.seq = QTTSeq(d, L, cond_dim=cond_dim, emb=128, hid=256, M=M, num_layers=2, dropout=0.1)

    def forward_sampled(self, cond, bits):
        """
        cond: [B, cond_dim]
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
