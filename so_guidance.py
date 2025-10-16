import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from ddpm_conditionals import TweediePosterior

class SecondOrderGUidedDiffusion:
    """
    Quadratic tweaked-proposal sampler.
    At step t, expand v_t at μ=μ_{t|t+1}(x_{t+1}):
        v_t(x) ≈ v_t(μ) + gᵀ(x-μ) + ½ (x-μ)ᵀ H (x-μ)
    Combined with base N(μ, σ_t^2 I) gives proposal N( μ + A^{-1}g, A^{-1} )
    where A = (1/σ_t^2) I - H.
    """
    def __init__(self, dist0, alpha_t, v, S=10_000, jitter=1e-5, max_jitter_tries=5):
        """
        dist0  : object with .score(x, alpha_bar) [and optionally .hess(x, alpha_bar)]
        alpha_t: array-like of α_t in (0,1]
        v      : callable v(t, x) -> (n,) for x shape (n,d), torch ops
        S      : batch size / #trajectories
        jitter : initial ridge added to precision A for PD
        """
        self.dist0   = dist0
        self.alpha_t = np.asarray(alpha_t, dtype=float)
        self.S       = int(S)
        self.d       = dist0.mu.shape[1]
        self.v       = v
        self.jitter  = float(jitter)
        self.max_jit = int(max_jitter_tries)

        self.alpha_bar = np.cumprod(self.alpha_t)

        # Wrap mixture scoring for TweediePosterior to (t, X)->...
        def score_t(t, X):
            return self.dist0.score(X, self.alpha_bar[t])
        self.tpost = TweediePosterior(
            alpha_bar_list=self.alpha_bar.tolist(),
            score=score_t,
            hess=(lambda t, X: self.dist0.hess(X, self.alpha_bar[t])) if hasattr(self.dist0, "hess") else None
        )

    def calc_samples(self):
        T = len(self.alpha_t)
        X = np.zeros((self.S, T, self.d))
        # Start from standard normal at t=T-1
        X[:, -1, :] = np.random.normal(size=(self.S, self.d))
        for t in tqdm(range(T - 2, -1, -1), desc="Second-order reverse"):
            X[:, t, :] = self.single_step(t, X[:, t + 1, :])
        return np.flip(X, axis=1)

    # @torch.no_grad()
    # def _sample_from_precision(self, mean: Tensor, A: Tensor) -> Tensor:
    #     """
    #     Given mean (S,d) and precision A (S,d,d), sample x ~ N(mean, A^{-1}).
    #     Uses Cholesky A = L Lᵀ, then x = mean + (Lᵀ)^{-1} z, z~N(0,I).
    #     """
    #     S, d = mean.shape
    #     L = None
    #     A_sym = 0.5 * (A + A.transpose(-1, -2))  # symmetrize
    #     eye = torch.eye(d, dtype=A.dtype, device=A.device).expand(S, d, d)

    #     jit = self.jitter
    #     for _ in range(self.max_jit):
    #         try:
    #             L = torch.linalg.cholesky(A_sym + jit * eye)
    #             break
    #         except RuntimeError:
    #             jit *= 10.0
    #     if L is None:
    #         # last resort: add strong ridge
    #         L = torch.linalg.cholesky(A_sym + (1e-2) * eye)

    #     z = torch.randn(S, d, dtype=mean.dtype, device=mean.device)
    #     # solve Lᵀ x_noise = z  => x_noise = (Lᵀ)^{-1} z = L^{-T} z
    #     x_noise = torch.linalg.solve_triangular(L.transpose(-1, -2), z, upper=True)
    #     return mean + x_noise

    def single_step(self, t: int, x_tplus1: np.ndarray) -> np.ndarray:
        """
        One reverse step using second-order (quadratic) tilt of v_t around μ_{t|t+1}(x_{t+1}).
        """
        # Base Gaussian mean μ and scalar variance σ_t^2
        mu_np = self.tpost.tweedie_posterior_mean_single_step(t, x_tplus1)  # (S,d)
        sigma2 = (1.0 - self.alpha_t[t]) / self.alpha_t[t]                  # σ_t^2

        # Compute grad and Hessian of v_t at μ (per sample)
        mu = torch.tensor(mu_np, dtype=torch.float32, requires_grad=True)
        t_vec = torch.full((mu.shape[0],), float(t), dtype=torch.float32)

        v_vals = self.v.predict_under_log(mu, t_vec, requires_grad = True)
        # gradient
        g = torch.autograd.grad(v_vals.sum(), mu, create_graph=True)[0]     # (S,d)

        # Hessian per sample (small d, so loop is fine)
        S, d = mu.shape
        H = torch.empty(S, d, d, dtype=mu.dtype)
        for i in range(S):
            # scalar function of x_i only
            def f_single(xi):
                return self.v.predict_under_log(xi.unsqueeze(0), t_vec[i:i+1], requires_grad=True).sum()
            Hi = torch.autograd.functional.hessian(f_single, mu[i], create_graph=False)
            H[i] = Hi

        # Precision A = (1/σ^2) I - H
        inv_sigma2 = float(1.0 / sigma2)
        I = torch.eye(d, dtype=mu.dtype)
        A = inv_sigma2 * I.unsqueeze(0).repeat(S, 1, 1) - H                 # (S,d,d)

        # Mean shift: solve A Δ = g  (no explicit inverse)
        # Use Cholesky solve (batched)
        A_sym = 0.5 * (A + A.transpose(-1, -2))
        # Stabilize with jitter for PD
        jitter = self.jitter
        solved = False
        for _ in range(self.max_jit):
            try:
                L = torch.linalg.cholesky(A_sym + jitter * I.unsqueeze(0))
                # cholesky_solve expects (S,d,1)
                Delta = torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)  # (S,d)
                solved = True
                break
            except RuntimeError:
                jitter *= 10.0
        if not solved:
            print("Warning: Cholesky failed, using ridge + direct solve")
            # fallback: ridge + direct solve
            A_num = A_sym + (1e-2) * I.unsqueeze(0)
            Delta = torch.linalg.solve(A_num, g)

        mean_tilted = mu + Delta                                             # (S,d)

        # Sample from N(mean_tilted, A^{-1}) without forming inverse
        # with torch.no_grad():
        #     x_t = self._sample_from_precision(mean_tilted, A_sym + jitter * I.unsqueeze(0))
        x_t = self._sample_from_precision(mean_tilted, A_sym)

        return x_t.cpu().numpy()
    
    @torch.no_grad()
    def _sample_from_precision(self, mean: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Given mean (S,d) and precision A (S,d,d), sample x ~ N(mean, A^{-1}).
        Uses Cholesky on the precision:
            A = L Lᵀ,  sample z~N(0,I),  x = mean + L^{-T} z
        Shape-safe even if A accidentally comes in as (d,d).
        """
        if mean.ndim != 2:
            raise ValueError(f"mean must be (S,d), got {mean.shape}")
        S, d = mean.shape

        # Ensure A is batched (S,d,d)
        if A.ndim == 2:
            A = A.unsqueeze(0).expand(S, d, d)
        elif A.ndim != 3 or A.shape[0] != S or A.shape[1:] != (d, d):
            raise ValueError(f"A must be (S,d,d) or (d,d), got {A.shape}, expected {(S,d,d)}")

        # Symmetrize for numerical stability
        A_sym = 0.5 * (A + A.transpose(-1, -2))

        # Make PD with a batched ridge if needed
        I = torch.eye(d, dtype=A.dtype, device=A.device).expand(S, d, d)
        jitter = self.jitter
        for _ in range(self.max_jit):
            try:
                L = torch.linalg.cholesky(A_sym + jitter * I)   # (S,d,d)
                break
            except RuntimeError:
                jitter *= 10.0
        else:
            # last resort
            L = torch.linalg.cholesky(A_sym + 1e-2 * I)

        # Sample via solve (batch-safe)
        z = torch.randn(S, d, 1, dtype=mean.dtype, device=mean.device)
        # Solve Lᵀ x_noise = z  -> x_noise = L^{-T} z
        x_noise = torch.linalg.solve_triangular(L.transpose(-1, -2), z, upper=True).squeeze(-1)  # (S,d)
        return mean + x_noise

