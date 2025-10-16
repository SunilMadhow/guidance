from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Tuple

Array = np.ndarray

def _ensure_2d(x: Array) -> Tuple[Array, bool]:
    """Ensure x is shape (n, d); return (x2d, squeezed_flag)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :], True
    if x.ndim != 2:
        raise ValueError(f"x must be (n,d) or (d,), got shape {x.shape}")
    return x, False

class TweediePosterior:
    """
    Tweedie-style posteriors for DDPM marginals.

    Args
    ----
    alpha_bar_list : list of float in (0, 1], cumulative ᾱ_t, assumed nonincreasing in t
    score : callable (t:int, x:(n,d)) -> (n,d)  giving s_t(x) = ∇_x log p_t(x)
    hess  : optional callable (t:int, x:(n,d)) -> (n,d,d) giving ∇²_x log p_t(x)
    """
    def __init__(
        self,
        alpha_bar_list: list[float],
        score: Callable[[int, Array], Array],
        hess: Optional[Callable[[int, Array], Array]] = None,
    ):
        self.alpha_bar_list = list(alpha_bar_list)
        self.score = score
        self.hess = hess

    # -------- x0 | xt  (Tweedie) --------
    def tweedie_posterior_mean(self, t: int, x_t: Array) -> Array:
        """
        μ_{0|t}(x_t) = (x_t + (1-ᾱ_t) s_t(x_t)) / √ᾱ_t
        """
        alpha_bar_t = float(self.alpha_bar_list[t])
        if not (0.0 < alpha_bar_t <= 1.0):
            raise ValueError(f"alpha_bar[{t}] must be in (0,1], got {alpha_bar_t}")
        X, squeezed = _ensure_2d(x_t)
        s = self.score(t, X)
        if s.shape != X.shape:
            raise ValueError(f"score(t={t},·) must return shape {X.shape}, got {s.shape}")
        mu = (X + (1.0 - alpha_bar_t) * s) / np.sqrt(alpha_bar_t)
        return mu[0] if squeezed else mu

    def tweedie_posterior_covariance(
        self,
        t: int,
        x_t: Array,
        hess_t: Optional[Callable[[Array], Array]] = None,
        fd_eps: float = 1e-3,
    ) -> Array:
        """
        Σ_{0|t}(x_t) = (1/ᾱ_t) [ (1-ᾱ_t) I + (1-ᾱ_t)^2 H_t(x_t) ],
        where H_t = ∇² log p_t. If no Hessian is supplied, uses
        finite-difference Jacobian of the score; if that's not possible,
        falls back to H_t = 0.
        """
        alpha_bar_t = float(self.alpha_bar_list[t])
        if not (0.0 < alpha_bar_t <= 1.0):
            raise ValueError(f"alpha_bar[{t}] must be in (0,1], got {alpha_bar_t}")

        X, squeezed = _ensure_2d(x_t)
        n, d = X.shape

        # 1) Prefer member hess, then per-call override, then FD score, else 0
        if self.hess is not None:
            H = self.hess(t, X)
            if H.shape != (n, d, d):
                raise ValueError(f"hess(t={t},·) must return {(n,d,d)}, got {H.shape}")
        elif hess_t is not None:
            H = hess_t(X)
            if H.shape != (n, d, d):
                raise ValueError(f"hess_t(·) must return {(n,d,d)}, got {H.shape}")
        else:
            # Finite-difference Hessian: H_ij = ∂ s_i / ∂ x_j
            if self.score is None:
                H = np.zeros((n, d, d), dtype=X.dtype)
            else:
                H = np.zeros((n, d, d), dtype=X.dtype)
                for j in range(d):
                    e = np.zeros((1, d), dtype=X.dtype)
                    e[0, j] = fd_eps
                    sp = self.score(t, X + e)
                    sm = self.score(t, X - e)
                    H[:, :, j] = (sp - sm) / (2.0 * fd_eps)

        I = np.eye(d, dtype=X.dtype)[None, :, :]
        one_minus = (1.0 - alpha_bar_t)
        Sigma = (1.0 / alpha_bar_t) * (one_minus * I + (one_minus**2) * H)
        return Sigma[0] if squeezed else Sigma

    # -------- xt | xt+1 (reverse mean) --------
    def tweedie_posterior_mean_single_step(self, t: int, x_tplus1: Array) -> Array:
        """
        μ_{t|t+1}(x_{t+1}) with α_{t+1} = ᾱ_{t+1} / ᾱ_t and score at t+1:
            μ = (x_{t+1} + (1-α_{t+1}) s_{t+1}(x_{t+1})) / √α_{t+1}
        Assumes ᾱ is nonincreasing so α_{t+1} ∈ (0,1].
        """
        if t + 1 >= len(self.alpha_bar_list):
            raise IndexError("t+1 exceeds length of alpha_bar_list.")
        alpha_bar_t   = float(self.alpha_bar_list[t])
        alpha_bar_tp1 = float(self.alpha_bar_list[t + 1])
        alpha_step    = alpha_bar_tp1 / alpha_bar_t
        if not (0.0 < alpha_step <= 1.0):
            raise ValueError(
                f"α_step = ᾱ[{t+1}]/ᾱ[{t}] = {alpha_bar_tp1}/{alpha_bar_t} = {alpha_step} not in (0,1]. "
                "Ensure alpha_bar_list is nonincreasing in t."
            )

        X, squeezed = _ensure_2d(x_tplus1)
        s = self.score(t + 1, X)
        if s.shape != X.shape:
            raise ValueError(f"score(t={t+1},·) must return shape {X.shape}, got {s.shape}")
        mu = (X + (1.0 - alpha_step) * s) / np.sqrt(alpha_step)
        return mu[0] if squeezed else mu

    

if __name__ == "__main__":
    d, n = 3, 5
    # ᾱ nonincreasing: e.g., [1.0, 0.7]
    alpha_bar = [1.0, 0.7]

    def score_dummy(t, x):   # grad log N(0, σ^2 I) at σ^2=2 => -x/2
        return -0.5 * x

    def hess_dummy(t, x):    # Hessian is constant -I/2
        I = np.eye(d)
        return np.tile((-0.5 * I)[None, :, :], (x.shape[0], 1, 1))

    x = np.random.randn(n, d)
    tp = TweediePosterior(alpha_bar_list=alpha_bar, score=score_dummy, hess=hess_dummy)

    mu0  = tp.tweedie_posterior_mean(0, x)
    Sig0 = tp.tweedie_posterior_covariance(0, x)
    mu_r = tp.tweedie_posterior_mean_single_step(0, x)
    print("Shapes:", mu0.shape, Sig0.shape, mu_r.shape)
