"""
FreqLens V4: Learnable Frequency Discovery with Axiomatic Attribution

Key Novelty (vs V1-V3):
1. Learnable Frequency Basis — discovers meaningful frequencies from data
   (no hardcoded physical frequencies → genuine knowledge discovery)
2. Axiomatic Attribution Framework — provably satisfies:
   - Completeness: ŷ_freq = Σ_f Attribution(f)
   - Faithfulness: M(F) - M(F\\{f}) = Attribution(f)
   - Null Frequency: amplitude(f)=0 ⟹ Attribution(f)=0
3. Shapley Equivalence Theorem: for the additive decomposition,
   per-frequency contribution = Shapley value (unique under axioms)
4. Frequency Diversity Regularization — prevents frequency collapse

Architecture:
  Input (B,L,C) → InputProj → (B,L,d)
    → LearnableFreqDecomp → coefficients (B,N,d), components (B,N,L,d)
    → SparseSelection (Gumbel top-K) → (B,K,d)
    → AxiomaticAttribution: contribution_f = MLP_f(c_f) → ŷ_freq = Σ_f contribution_f
    → ResidualPath: ŷ_res = MLP(flatten(x))
    → Fusion: ŷ = α·ŷ_freq + (1-α)·ŷ_res

Theory: See Section 3.4 of the paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List


# ---------------------------------------------------------------------------
# 1. Learnable Frequency Decomposition
# ---------------------------------------------------------------------------
class LearnableFrequencyDecomposition(nn.Module):
    """
    Learnable Frequency Decomposition Layer.

    Instead of using hardcoded physical frequencies (daily, weekly, yearly),
    this module **learns** the frequency bases from data.  After training,
    the discovered frequencies can be mapped back to physical periods for
    interpretation — constituting *genuine* knowledge discovery.

    Mathematical formulation:
        basis_i(t) = cos(2π · f_i · t + φ_i)        i = 1, ..., N
        c_i = ⟨x, basis_i⟩ / ‖basis_i‖²         (projection coefficient)
        component_i(t) = c_i · basis_i(t)           (per-frequency component)
        x̂(t) = Σ_i component_i(t)                  (reconstruction)

    Parameters:
        seq_len:       Input sequence length
        n_channels:    Hidden dimension (d_model)
        n_freq_bases:  Number of learnable frequency bases
        freq_init:     Initialization strategy ('log_uniform', 'linear', 'random')
    """

    def __init__(
        self,
        seq_len: int,
        n_channels: int,
        n_freq_bases: int = 32,
        freq_init: str = "log_uniform",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.n_freq_bases = n_freq_bases

        # ----- learnable frequencies (sigmoid-parameterized → bounded positive) -----
        # frequencies = min_freq + (max_freq - min_freq) * sigmoid(raw_param)
        # To init at target freq f: raw = logit((f - min_freq) / (max_freq - min_freq))
        min_freq = 1.0 / (seq_len * 10)
        max_freq = 0.5

        if freq_init == "log_uniform":
            # Log-uniform target freqs from 1/seq_len to 0.5 (Nyquist)
            target_freqs = torch.logspace(
                math.log10(1.0 / seq_len), math.log10(0.5), n_freq_bases
            )
        elif freq_init == "linear":
            target_freqs = torch.linspace(1.0 / seq_len, 0.5, n_freq_bases)
        else:  # random
            target_freqs = torch.empty(n_freq_bases).uniform_(1.0 / seq_len, 0.5)

        # Convert target freqs → logits for the sigmoid parameterization
        # sigmoid(x) = (f - min_freq) / (max_freq - min_freq) → x = logit(...)
        t = (target_freqs - min_freq) / (max_freq - min_freq)
        t = t.clamp(1e-6, 1 - 1e-6)  # numerical safety for logit
        init_logits = torch.log(t / (1 - t))  # inverse sigmoid = logit

        self.log_frequencies = nn.Parameter(init_logits)

        # ----- learnable phases per frequency -----
        self.phases = nn.Parameter(torch.zeros(n_freq_bases))

        # ----- time index buffer -----
        self.register_buffer(
            "time_indices", torch.arange(seq_len, dtype=torch.float32)
        )

    # ---- properties --------------------------------------------------------
    @property
    def frequencies(self) -> torch.Tensor:
        """Positive frequencies via sigmoid mapping to [min_freq, Nyquist].

        Uses sigmoid instead of exp+clamp to avoid dead gradients at boundaries.
        The log_frequencies parameter is treated as a raw logit that gets mapped
        through sigmoid into the valid frequency range.
        """
        min_freq = 1.0 / (self.seq_len * 10)  # max observable period
        max_freq = 0.5                          # Nyquist frequency
        return min_freq + (max_freq - min_freq) * torch.sigmoid(self.log_frequencies)  # (N,)

    @property
    def periods(self) -> torch.Tensor:
        """Physical periods = 1 / frequency."""
        return 1.0 / (self.frequencies + 1e-10)

    # ---- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose *x* into learnable frequency components.

        Args:
            x: (B, L, D)  — projected hidden features

        Returns:
            dict with keys:
                coefficients    (B, N, D)  — projection coefficients
                freq_components (B, N, L, D) — per-frequency reconstruction
                reconstruction  (B, L, D)  — sum of all components
                frequencies     (N,)       — current learned frequencies
                basis           (N, L)     — raw cosine bases
                basis_norm      (N, L)     — L2-normalised bases
        """
        B, L, D = x.shape
        freqs = self.frequencies  # (N,)
        phases = self.phases      # (N,)
        t = self.time_indices[:L] # (L,)

        # basis_i(t) = cos(2π f_i t + φ_i)  →  shape (N, L)
        angle = (
            2.0 * math.pi * freqs[:, None] * t[None, :] + phases[:, None]
        )
        basis = torch.cos(angle)  # (N, L)

        # L2-normalise each basis vector (for proper orthogonal projection)
        basis_norm_factor = torch.norm(basis, dim=1, keepdim=True) + 1e-8
        basis_norm = basis / basis_norm_factor  # (N, L)

        # projection coefficient: c_i = <x, basis_i> / ||basis_i||
        # x: (B,L,D), basis_norm: (N,L) → coefficients: (B,N,D)
        coefficients = torch.einsum("bld, nl -> bnd", x, basis_norm)

        # per-frequency component: component_i(t) = c_i · basis_norm_i(t)
        freq_components = torch.einsum(
            "bnd, nl -> bnld", coefficients, basis_norm
        )  # (B, N, L, D)

        # reconstruction
        reconstruction = freq_components.sum(dim=1)  # (B, L, D)

        return {
            "coefficients": coefficients,       # (B, N, D)
            "freq_components": freq_components,  # (B, N, L, D)
            "reconstruction": reconstruction,    # (B, L, D)
            "frequencies": freqs,                # (N,)
            "basis": basis,                      # (N, L)
            "basis_norm": basis_norm,             # (N, L)
        }

    # ---- regularisation losses --------------------------------------------
    def diversity_loss(self) -> torch.Tensor:
        """
        Frequency diversity regularisation (log-barrier on consecutive gaps
        in log-frequency space).  Prevents frequency collapse.

        L_div = −(1/M) Σ_{i} log(Δ_i + ε)

        where Δ_i = log f_{(i+1)} − log f_{(i)} on the sorted actual frequencies.
        """
        sorted_freqs, _ = torch.sort(self.frequencies)
        sorted_log_freqs = torch.log(sorted_freqs + 1e-10)
        gaps = sorted_log_freqs[1:] - sorted_log_freqs[:-1]  # (N-1,)
        loss = -torch.log(gaps + 1e-8).mean()
        return loss

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """MSE between input and sum-of-components reconstruction."""
        out = self.forward(x)
        return F.mse_loss(out["reconstruction"], x)

    # ---- interpretation utilities -----------------------------------------
    def get_period_table(self, time_unit: str = "steps") -> List[Dict]:
        """
        Map learned frequencies → physical periods (for paper tables).

        Args:
            time_unit: base unit name (for display only)

        Returns:
            list of dicts  [{freq_idx, frequency, period, period_str}, ...]
        """
        freqs = self.frequencies.detach().cpu().numpy()
        periods = 1.0 / (freqs + 1e-10)
        table = []
        for i, (f, p) in enumerate(zip(freqs, periods)):
            table.append(
                {
                    "freq_idx": i,
                    "frequency": float(f),
                    "period": float(p),
                    "period_str": f"{p:.1f} {time_unit}",
                }
            )
        return sorted(table, key=lambda d: d["period"])


# ---------------------------------------------------------------------------
# 2. Sparse Frequency Selection  (Gumbel-Softmax Top-K)
# ---------------------------------------------------------------------------
class SparseFrequencySelector(nn.Module):
    """
    Differentiable top-K frequency selection via Gumbel-Softmax.

    During training: soft selection with temperature annealing (τ: 1→0.1).
    During inference: hard argmax.

    Args:
        n_freqs:    Total number of frequency bases
        d_model:    Feature dimension (for computing importance logits)
        top_k:      Number of frequencies to keep
    """

    def __init__(
        self,
        n_freqs: int,
        d_model: int,
        top_k: int = 8,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
    ):
        super().__init__()
        self.n_freqs = n_freqs
        self.d_model = d_model
        self.top_k = top_k
        self.tau_init = tau_init
        self.tau_min = tau_min

        # lightweight scorer: coefficient vector → scalar importance per freq
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # learnable prior (bias)
        self.base_logits = nn.Parameter(torch.zeros(n_freqs))

    def forward(
        self,
        coefficients: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            coefficients: (B, N, D) — projection coefficients from decomp
            temperature:  Gumbel τ (None → use default schedule)

        Returns:
            selected_coeffs: (B, K, D) — selected frequency coefficients
            selection_mask:   (B, N)   — binary mask
            importance:       (B, N)   — normalised importance scores
            selected_indices: (B, K)   — indices of selected frequencies
        """
        B, N, D = coefficients.shape
        tau = temperature if temperature is not None else self.tau_init

        # per-frequency importance logit
        logits = self.scorer(coefficients).squeeze(-1)  # (B, N)
        logits = logits + self.base_logits.unsqueeze(0)  # add prior

        if self.training:
            # Gumbel noise for exploration
            u = torch.rand_like(logits).clamp(1e-10, 1.0 - 1e-10)
            gumbel = -torch.log(-torch.log(u))
            perturbed = (logits + gumbel) / tau
            probs = F.softmax(perturbed, dim=-1)  # (B, N)
        else:
            probs = F.softmax(logits, dim=-1)

        # top-k selection
        _, indices = torch.topk(probs, self.top_k, dim=-1)  # (B, K)

        # hard mask
        mask = torch.zeros_like(probs)
        mask.scatter_(1, indices, 1.0)

        # gather selected coefficients
        batch_idx = (
            torch.arange(B, device=coefficients.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
        )
        selected = coefficients[batch_idx, indices]  # (B, K, D)

        # normalised importance for interpretability
        importance = F.softmax(logits, dim=-1)  # (B, N)

        return selected, mask, importance, indices


# ---------------------------------------------------------------------------
# 3. Axiomatic Attribution Head
# ---------------------------------------------------------------------------
class AxiomaticAttributionHead(nn.Module):
    """
    Attribution Prediction Head with axiomatic guarantees.

    Design principle — **strict additive decomposition**:
        ŷ_freq = Σ_{f ∈ S} contribution(f)       (Completeness axiom)
        contribution(f) = MLP_f(c_f)              (independent per frequency)

    Axioms satisfied (by construction):
      A1  Completeness    Σ_f A(f) = ŷ_freq
      A2  Faithfulness    M(S) − M(S\\{f}) = A(f) = contribution(f)
      A3  Null frequency  c_f = 0 ⟹ contribution(f) ≈ 0
      A4  Symmetry        c_i = c_j ∧ same MLP ⟹ A(i) = A(j)

    Theorem (Shapley equivalence):
      For an additive value function v(S) = Σ_{f∈S} contribution(f),
      the Shapley value φ_f = contribution(f).  Hence the per-frequency
      contribution IS the unique attribution satisfying the four Shapley
      axioms (Efficiency, Symmetry, Null-player, Linearity).
    """

    def __init__(
        self,
        n_freqs: int,
        d_model: int,
        pred_len: int,
        n_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_freqs = n_freqs
        self.d_model = d_model
        self.pred_len = pred_len
        self.n_channels = n_channels

        # independent MLP per frequency — *no* shared weights
        # (shared weights would introduce implicit interactions → violate A2)
        # No bias terms to ensure Null axiom (A3) holds by construction
        self.freq_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model, bias=False),  # No bias for A3
                    nn.ReLU(),  # ReLU ensures MLP(0) = 0
                    nn.Dropout(dropout),
                    nn.Linear(d_model, pred_len * n_channels, bias=False),  # No bias for A3
                )
                for _ in range(n_freqs)
            ]
        )

        # Initialize weights
        for proj in self.freq_projectors:
            nn.init.xavier_uniform_(proj[0].weight, gain=0.01)
            nn.init.xavier_uniform_(proj[-1].weight, gain=0.01)

    def forward(
        self, freq_coefficients: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            freq_coefficients: (B, K, D) — selected frequency coefficients

        Returns:
            prediction:    (B, pred_len, C)            = Σ_f contribution_f
            contributions: (B, K, pred_len, C)         per-freq contributions
            attr_dict:     percentages, norms, etc.
        """
        B, K, D = freq_coefficients.shape

        contribs = []
        for i in range(min(K, self.n_freqs)):
            c = self.freq_projectors[i](freq_coefficients[:, i, :])
            c = c.view(B, self.pred_len, self.n_channels)
            contribs.append(c)

        # (B, K, pred_len, C)
        contributions = torch.stack(contribs, dim=1)

        # ---- STRICT ADDITIVE SUM (Completeness Axiom) ----
        prediction = contributions.sum(dim=1)  # (B, pred_len, C)

        # ---- attribution percentages (for visualisation) ----
        contrib_norms = torch.norm(
            contributions.detach(), dim=(2, 3)
        )  # (B, K)
        total_norm = contrib_norms.sum(dim=1, keepdim=True) + 1e-8
        percentages = contrib_norms / total_norm  # (B, K)

        attr_dict = {
            "percentages": percentages,
            "contribution_norms": contrib_norms,
        }

        return prediction, contributions, attr_dict

    # ---- axiomatic verification ------------------------------------------
    def verify_completeness(
        self,
        prediction: torch.Tensor,
        contributions: torch.Tensor,
        atol: float = 1e-5,
    ) -> bool:
        """Verify Completeness axiom: ŷ == Σ_f contribution_f."""
        recon = contributions.sum(dim=1)
        diff = (prediction - recon).abs().max().item()
        return diff < atol

    def verify_faithfulness(
        self,
        freq_coefficients: torch.Tensor,
        freq_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Verify Faithfulness axiom for frequency *freq_idx*:
            M(S) − M(S\\{f}) == contribution(f)

        Returns:
            (full_pred, ablated_pred) so caller can check the difference.
        """
        full_pred, contributions, _ = self.forward(freq_coefficients)
        # ablate: zero out freq_idx
        ablated = freq_coefficients.clone()
        ablated[:, freq_idx, :] = 0.0
        ablated_pred, _, _ = self.forward(ablated)
        # difference should equal contributions[:, freq_idx]
        return full_pred, ablated_pred


# ---------------------------------------------------------------------------
# 4. FreqLens V4 — Main Model
# ---------------------------------------------------------------------------
class FreqLensV4(nn.Module):
    """
    FreqLens V4: Learnable Frequency Discovery + Axiomatic Attribution.

    Architecture:
        x_enc (B,L,C)
          → input_proj → h (B,L,d_model)
          → LearnableFreqDecomp → coeff (B,N,d), components (B,N,L,d)
          → SparseSelector (Gumbel top-K) → selected_coeff (B,K,d)
          → AxiomaticAttribution → ŷ_freq = Σ_f contribution_f
          → ResidualPath → ŷ_res = MLP(flatten(x))
          → ŷ = α·ŷ_freq + (1−α)·ŷ_res

    Training loss:
        L = L_pred + λ_div·L_diversity + λ_recon·L_recon + λ_sparse·L_sparse
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        enc_in: int = 1,
        d_model: int = 64,
        n_freq_bases: int = 32,
        top_k: int = 8,
        dropout: float = 0.1,
        freq_init: str = "log_uniform",
        use_gumbel_softmax: bool = True,
        gumbel_tau: float = 1.0,
        # --- Ablation flags ---
        freeze_frequencies: bool = False,
        no_residual: bool = False,
        shared_attribution: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.n_freq_bases = n_freq_bases
        self.top_k = min(top_k, n_freq_bases)
        self.use_gumbel_softmax = use_gumbel_softmax
        self.gumbel_tau = gumbel_tau
        self.no_residual = no_residual
        self.shared_attribution = shared_attribution

        # 1. input projection
        self.input_proj = nn.Linear(enc_in, d_model)

        # 2. learnable frequency decomposition
        self.freq_decomp = LearnableFrequencyDecomposition(
            seq_len=seq_len,
            n_channels=d_model,
            n_freq_bases=n_freq_bases,
            freq_init=freq_init,
        )

        # Ablation: freeze frequency parameters (no learning, stay at init)
        if freeze_frequencies:
            self.freq_decomp.log_frequencies.requires_grad_(False)
            self.freq_decomp.phases.requires_grad_(False)

        # 3. sparse frequency selection
        self.sparse_select = SparseFrequencySelector(
            n_freqs=n_freq_bases,
            d_model=d_model,
            top_k=self.top_k,
        )

        # 4. axiomatic attribution head (or shared-MLP ablation variant)
        if shared_attribution:
            # Ablation: single shared MLP for all frequencies (violates axiom A2)
            self.shared_head = nn.Sequential(
                nn.Linear(d_model * self.top_k, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, pred_len * enc_in),
            )
            self.attribution_head = None
        else:
            self.attribution_head = AxiomaticAttributionHead(
                n_freqs=self.top_k,
                d_model=d_model,
                pred_len=pred_len,
                n_channels=enc_in,
                dropout=dropout,
            )
            self.shared_head = None

        # 5. residual path  (captures non-periodic components)
        if not no_residual:
            self.residual_proj = nn.Sequential(
                nn.Linear(seq_len * enc_in, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, pred_len * enc_in),
            )
            # learnable fusion weight  α ∈ (0,1)
            self.raw_alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        else:
            self.residual_proj = None
            self.raw_alpha = None

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    # ---- initialisation ---------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---- temperature schedule ---------------------------------------------
    def get_gumbel_temperature(
        self, epoch: int, total_epochs: int
    ) -> float:
        min_tau, max_tau = 0.1, 1.0
        tau = max_tau - (max_tau - min_tau) * min(epoch / total_epochs, 1.0)
        return max(tau, min_tau)

    # ---- forward ----------------------------------------------------------
    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        y_dec: Optional[torch.Tensor] = None,
        y_mark_dec: Optional[torch.Tensor] = None,
        gumbel_temperature: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_enc: (B, L, C) — input time series

        Returns:
            dict with keys:
                predictions          (B, pred_len, C)
                freq_prediction      (B, pred_len, C)
                residual_prediction  (B, pred_len, C)
                contributions        (B, K, pred_len, C)
                attributions         (B, N)   — full attribution vector
                learned_frequencies  (N,)     — current frequency values
                learned_periods      (N,)     — current period values
                selected_indices     (B, K)
                importance_scores    (B, N)
                selection_mask       (B, N)
                alpha                scalar   — fusion weight
                reconstruction       (B, L, d_model)
                coefficients         (B, N, d_model)
        """
        B, L, C = x_enc.shape

        # 1. project to hidden dim
        h = self.input_proj(x_enc)   # (B, L, d_model)
        h = self.dropout(h)

        # 2. learnable frequency decomposition
        decomp = self.freq_decomp(h)
        coefficients = decomp["coefficients"]       # (B, N, d)
        freq_components = decomp["freq_components"] # (B, N, L, d)
        reconstruction = decomp["reconstruction"]   # (B, L, d)

        # 3. sparse selection
        selected_coeffs, mask, importance, indices = self.sparse_select(
            coefficients, temperature=gumbel_temperature,
        )  # (B,K,d), (B,N), (B,N), (B,K)

        # 4. prediction path (axiomatic attribution or shared-MLP ablation)
        if self.shared_attribution and self.shared_head is not None:
            # Ablation: shared MLP head (no per-frequency independence)
            freq_pred = self.shared_head(
                selected_coeffs.reshape(B, -1)
            ).view(B, self.pred_len, C)
            contributions = None
            attr_dict = {"percentages": torch.zeros(B, self.top_k, device=x_enc.device)}
        else:
            freq_pred, contributions, attr_dict = self.attribution_head(
                selected_coeffs,
            )  # (B,pred,C), (B,K,pred,C), dict

        # 5. residual path (disabled in no_residual ablation)
        if self.no_residual or self.residual_proj is None:
            predictions = freq_pred
            residual_pred = torch.zeros_like(freq_pred)
            alpha_val = 1.0
        else:
            residual_pred = self.residual_proj(
                x_enc.reshape(B, -1)
            ).view(B, self.pred_len, C)
            alpha = torch.sigmoid(self.raw_alpha)
            alpha_val = alpha.item()
            predictions = alpha * freq_pred + (1.0 - alpha) * residual_pred

        # ---- build full-dimensional attribution vector --------------------
        attr_pct = attr_dict["percentages"]          # (B, K)
        full_attr = torch.zeros(B, self.n_freq_bases, device=x_enc.device)
        batch_idx = (
            torch.arange(B, device=x_enc.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
        )
        full_attr[batch_idx, indices] = attr_pct

        return {
            "predictions": predictions,
            "freq_prediction": freq_pred,
            "residual_prediction": residual_pred,
            "contributions": contributions,
            "attributions": full_attr,
            "attr_percentages": attr_pct,
            "learned_frequencies": decomp["frequencies"],
            "learned_periods": self.freq_decomp.periods,
            "selected_indices": indices,
            "importance_scores": importance,
            "selection_mask": mask,
            "alpha": alpha_val,
            "reconstruction": reconstruction,
            "coefficients": coefficients,
        }

    # ---- auxiliary losses (called by trainer) -----------------------------
    def diversity_loss(self) -> torch.Tensor:
        """Frequency diversity regularisation."""
        return self.freq_decomp.diversity_loss()

    def reconstruction_loss(self, x_hidden: torch.Tensor) -> torch.Tensor:
        """Reconstruction quality of the frequency decomposition."""
        return self.freq_decomp.reconstruction_loss(x_hidden)

    # ---- interpretability utilities ---------------------------------------
    def get_discovered_periods(self, time_unit: str = "steps") -> List[Dict]:
        """Return learned frequencies as a human-readable period table."""
        return self.freq_decomp.get_period_table(time_unit=time_unit)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
