Neural MPS Module – User Guide
================================

Focus: (1) Model architecture, (2) Dataset creation system, (3) Semi‑supervised training process.

Audience: You can code, but you may be new to ML / tensor networks. This document is intentionally explicit about data shapes and flow.

---
## 1. Problem & Representation
We learn a parametric family of smooth d‑dimensional functions

    f_A,c,Σ (x) = A * exp( -1/2 (x - c)^T Σ^{-1} (x - c) )

where parameters are:
* A ∈ [0.2, 1.0]
* c ∈ R^d (each component in [-0.5, 0.5])
* Σ is a d×d covariance matrix built from random standard deviations and a prescribed correlation level.

Instead of storing the full N^d grid of values, we approximate (or predict) the function as a Matrix Product State (MPS), also known as a Tensor Train (TT), or a Binary / Quantized variant (BTT/QTT style) that factorises the tensor into a chain of small cores.

For a TT with physical size n (same n in each dimension for simplicity), ranks r_0=1, r_1, …, r_{d-1}, r_d=1, each core has shape [r_{i}, n, r_{i+1}]. The full tensor value at a multi‑index (x_1,…,x_d) is the contracted product of slices core_i[:, x_i, :].

Binary TT (BTT) replaces each physical index of size N (assumed power of two: N = 2^k) by k binary sites of size 2, lengthening the chain from d to d*k sites but reducing each n_i to 2. This can yield lower parameter counts when N is large.

---
## 2. Dataset Creation System (`data_processing/create_dataset.py`)
### 2.1 Parameter Vector Assembly
For each synthetic sample we store a parameter vector (PyTorch 1D tensor of shape `[input_size]`) constructed as:

1. Amplitude A (1 value)
2. Center c (d values)
3. Upper triangular (including diagonal) of covariance matrix Σ – that’s d(d+1)/2 values

Total: 1 + d + d(d+1)/2 = input_size (this is written into `info.json`).

### 2.2 Generating Targets
Depending on `--semi-supervised`:
* Supervised = build a *TT approximation* of the full d‑dimensional function using cross approximation (`tn.cross`). Output target is a Python list of d (TT) or d*k (BTT) core tensors: each core_i has shape `[r_i, n_i, r_{i+1}]`.
* Semi‑supervised = (current prototype) store *raw function evaluations* along a simple 1D grid (`grid = linspace(-1,1,N)`); each sample target is a 1D tensor of length N (intended as a reduced supervision signal). The semi‑supervised training script is partially scaffolded and will evolve to map these 1D slices to a latent TT via indirect loss.

### 2.3 TT vs BTT Rank Pattern
* TT: ranks list is `[1, max_rank, max_rank, …, max_rank, 1]` (length d+1).
* BTT: Let total sites = d*k. The code builds a *“rank pyramid”*: start at 1, double (capped at `max_rank`) until halfway, mirror back down (palindromic). Example: d=4, N=64 ⇒ k=6 (since 2^6=64), sites=24. Ranks might grow 1→2→4→8→16→20 (capped) then mirror back to 1.

### 2.4 Splits & Files
Data is split (80/10/10) into `train.pt`, `val.pt`, `test.pt`, plus `info.json` with metadata (format, d, N, ranks, correlation, etc.).

### 2.5 Supervised vs Semi‑Supervised Summary
| Aspect | Supervised | Semi‑Supervised |
|--------|------------|-----------------|
| Target stored | Full list of TT/BTT cores | Lower‑dimensional function samples (prototype: 1D grid) |
| Loss | Direct MSE per predicted core | (Planned) MSE on reconstructed function samples after contracting predicted cores | 
| Pros | Strong signal, fast convergence | Lower storage, scalable when full TT expensive |
| Cons | Larger disk footprint; requires TT construction | Indirect supervision; needs reconstruction logic |

Command examples:

    # Supervised TT example
    python create_dataset.py --n-samples 10000 --d 4 --N 64 --max-rank 20 --format TT --semi-supervised False

    # Semi-supervised BTT example
    python create_dataset.py --n-samples 10000 --d 4 --N 64 --max-rank 20 --format BTT --semi-supervised True

---
## 3. Model Architecture (`model/neural_mps.py`)
High level pipeline:

    params (batch, input_size)
      → Fully connected trunk → latent (batch, latent_size)
      → Decoder (shared or split) → list of TT/BTT cores (batch, r_i, n_i, r_{i+1})

### 3.1 Trunk (Fully Connected Layers)
Arguments: `hidden_size`, `num_layers`, `latent_size`, `dropout`.

Flow (example numbers): Suppose d=4, N=64 (TT), ranks = [1, 8, 12, 6, 1], input_size = 1 + 4 + 4*5/2 = 15.
1. Input: (B, 15)
2. fc1 → ReLU → Dropout: (B, hidden_size)
3. (num_layers-1) additional hidden FCLayer blocks: each keeps (B, hidden_size)
4. fc2 → latent_size (e.g. 64): output (B, 64)

### 3.2 Decoder Types
#### A. Shared Decoder
Goal: produce a *single large feature map* then slice sub‑blocks to assemble all cores.

Steps:
1. Linear projection: (B, latent) → (B, channel_size * H * W); with H=W=max(ranks) for simplicity.
2. Reshape: (B, channel_size, H, W)
3. Two ConvLayer blocks (Conv2d + ReLU + Dropout): shape preserved.
4. Final 2D conv to produce channel dimension = n (physical size). Result: (B, n, H, W).
5. For each core i: slice the top‑left `[ : , : , 0:r_i , 0:r_{i+1} ]` giving (B, n, r_i, r_{i+1}); then permute to (B, r_i, n, r_{i+1}).

Example slice (ranks [1,8,12,6,1], n=64, H=W=12):
* Core 0: (B, 64, 1, 8) → (B, 1, 64, 8)
* Core 1: (B, 64, 8, 12) → (B, 8, 64, 12)
* Core 2: (B, 64, 12, 6) → (B, 12, 64, 6)
* Core 3: (B, 64, 6, 1) → (B, 6, 64, 1)

Advantages: Shared computation across cores; parameter efficient for many cores.

#### B. Split Decoder (Per‑Core `CoreDecoder`)
Each core has its own small decoder starting from the *same* latent vector.

CoreDecoder pipeline for one core with dims (r_i, n_i, r_{i+1}):
1. FC layer (latent → latent) with nonlinearity + dropout.
2. Reshape to (B, latent, 1, 1).
3. A stack of ConvTranspose (upsampling) layers doubling spatial size each time until reaching ≥ max(n_i, r_{i+1}). Number of stages = ceil(log2(max(n_i, r_{i+1}))).
4. Adaptive average pool → (B, latent', n_i, r_{i+1}).
5. Final 2D conv mapping channels latent' → r_i, yielding (B, r_i, n_i, r_{i+1}).

Concrete example (from `train/test.py` style): ranks [1,5,6,4,1], n=10.
* Target core dims e.g. second core: r_i=5, n_i=10, r_{i+1}=6 ⇒ need spatial (10,6). max(10,6)=10 ⇒ stages = ceil(log2 10)=4.
* Stages sizes (square conceptual since stride=2): 1→2→4→8→16 (after 4 layers we exceed 10). After adaptive pool: (10,6). Final conv → (B,5,10,6).

Advantages: Independent per core capacity; flexible when ranks differ widely. Trade‑off: More parameters and compute.

### 3.3 Forward Output Contracting
To evaluate a predicted TT for index tuple (x_1,…,x_d), slice each core: core_i_slice = core_i[ :, :, x_i, : ] giving shape (B, r_i, r_{i+1}). Left‑to‑right batched matrix multiply collapses to (B, 1, 1) = scalar value per sample.

Provided utilities (`eval_tt`, `eval_btt`) implement this contraction for TT and binary TT respectively.

---
## 4. Training
### 4.1 Supervised (`train/supervised_train.py`)
Loop:
1. Load (params, cores_true_list)
2. Predict cores_pred_list = model(params)
3. Loss = mean over cores of MSE(core_pred, core_true)
4. Optimiser: AdamW with weight decay
5. Early stopping after an offset epoch count: track best validation loss with patience & delta.

Why average across cores? Prevents one large core dominating and stabilises scale across varying ranks.

Checkpointing: On exception saves partial state. Best model parameters (based on validation) restored at end.

### 4.2 Early Stopping Logic
* Offset: ignore first few epochs (warm‑up)
* Improvement criterion: new_val_loss < best_loss - delta
* Patience: stop after consecutive non‑improvements.

### 4.3 Semi‑Supervised (`train/semi_supervised_train.py` – WIP)
Current scaffold mirrors supervised training but leaves TODOs for:
* Computing predictions (cores) from params
* Sampling index tuples or reconstructing function slices
* Forming a loss against stored lower‑dimensional targets

Intended idea:
1. Predict cores.
2. Sample a mini‑set of multi‑indices (or binary indices for BTT).
3. Contract predicted cores via `eval_tt` / `eval_btt`.
4. Compare against ground truth scalar values derived from the stored semi‑supervised target (or by re‑evaluating analytic function if accessible) using MSE.
5. Optionally regularise core norms or add rank sparsity penalties.

This reduces dataset size: you store O(N) instead of O(d * r^2 * n) per sample.

---
## 5. Practical Choices & Tips
* Choose `shared` decoder when d (or d*k) is large; choose `split` when ranks vary or you need per‑core flexibility.
* For BTT ensure N is a power of two; code assumes N = 2^k.
* Monitor both per‑core losses (add logging if debugging) to detect any systematic mismatch in a particular core—often indicates rank insufficiency.
* If semi‑supervised training underfits, consider hybrid: pretrain semi‑supervised, fine‑tune supervised on a smaller set of fully constructed TT cores.

---
## 6. File Reference
| File | Purpose |
|------|---------|
| `data_processing/create_dataset.py` | Synthetic data & TT/BTT generation |
| `model/neural_mps.py` | Trunk + decoders predicting TT cores |
| `model/layers.py` | Reusable FC / Conv / Deconv blocks |
| `train/supervised_train.py` | Full core supervision training loop |
| `train/semi_supervised_train.py` | WIP semi‑supervised framework |
| `train/utils.py` | Logging, folder creation, dataset discovery (not detailed here) |

---
## 7. Extending
* Add positional encoding of parameters if function family broadens.
* Swap MSE for a relative error metric (scale invariance) if A varies widely.
* Implement adaptive rank growth: start low and increase ranks when validation plateaus.
* Semi‑supervised: store a *set of sampled multi‑indices with values* instead of a 1D grid to better approximate diverse regions.

---
## 8. Glossary
* TT / MPS: Tensor factorisation representing large tensors via chain of cores.
* Rank: Internal dimension coupling adjacent cores (controls expressiveness).
* BTT / QTT: Variant using binary expansion of indices to reduce per‑site dimension.
* Core: 3D tensor (r_i, n_i, r_{i+1}) capturing local factor interaction.

---
## 9. Minimal Usage Sequence (Supervised)
1. Generate dataset.
2. Run supervised training script with matching format (TT or BTT).
3. Load `best_model.pth` and call `model(params_batch)` to get predicted cores.
4. Use `eval_tt` / `eval_btt` to evaluate function values at indices.

---
Questions / improvements: see semi‑supervised TODOs—feel free to refine contraction sampling & loss shaping.
