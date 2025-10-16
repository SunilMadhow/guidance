import numpy as np
import torch
from tqdm import tqdm
from ddpm_conditionals import TweediePosterior

class LinearizedDiffusion:
    """
    Linearized tweaked-proposal sampler.
    At step t: base kernel N(μ_{t|t+1}(x_{t+1}), σ_t^2 I) is tilted by linearizing v_t at μ,
    yielding N( μ + σ_t^2 ∇v_t(μ), σ_t^2 I ).  (See “Tweaked proposals”.)
    """
    def __init__(self, dist0, alpha_t, v, S=10_000):
        """
        dist0  : object with .score(x, alpha_bar) [and optionally .hess(x, alpha_bar)]
        alpha_t: array-like of step multipliers α_t in (0,1]
        v      : callable v(t, x) returning shape (n,) for x shape (n,d)
        S      : batch size / number of trajectories
        """
        self.dist0     = dist0
        self.alpha_t   = np.asarray(alpha_t, dtype=float)
        self.S         = int(S)
        self.d         = dist0.mu.shape[1]
        self.v         = v
        self.alpha_bar = np.cumprod(self.alpha_t)  # <- no trailing comma

        # Wrap mixture score/hessian to signature (t, X) -> ...
        def score_t(t, X):
            return self.dist0.score(X, self.alpha_bar[t])
        def hess_t(t, X):
            if hasattr(self.dist0, "hess"):
                return self.dist0.hess(X, self.alpha_bar[t])
            # If TweediePosterior supports FD fallback, return None; otherwise omit hess.
            return None

        self.tpost = TweediePosterior(
            alpha_bar_list=self.alpha_bar.tolist(),
            score=score_t,
            hess=(hess_t if hasattr(self.dist0, "hess") else None)
        )

    def calc_samples(self):
        T = len(self.alpha_t)
        X = np.zeros((self.S, T, self.d))
        # start from standard normal at t = T-1
        X[:, -1, :] = np.random.normal(size=(self.S, self.d))
        for t in tqdm(range(T - 2, -1, -1), desc="Linearized reverse"):
            X[:, t, :] = self.single_step(t, X[:, t + 1, :])
        # return in chronological order (x_0 ... x_T-1)
        return np.flip(X, axis=1)
    


    def single_step(self, t, x):
        """
        One reverse step using a linearized tilt of v_t(·) around μ_t(x_{t+1}),
        matched to Diffusion.single_step.
        """
        # --- Base reverse Gaussian: match Diffusion exactly ---
        alpha_t = self.alpha_t[t]
        alpha_bar_t = self.alpha_bar[t]
        score = self.dist0.score(x, alpha_bar_t)                  # s_t(x_{t+1})
        mu    = (x + (1.0 - alpha_t) * score) / np.sqrt(alpha_t)  # μ_t
        var   = (1.0 - alpha_t) / alpha_t                                # σ_t^2
        std   = np.sqrt(var)

        # --- Gradient g = ∇_x v_t(x) evaluated at x=μ ---
        mu_torch = torch.tensor(mu, dtype=torch.float32, requires_grad=True)
        T = len(self.alpha_t)
        t_torch = torch.full((mu.shape[0],), float(T - 1 - t), dtype=torch.float32)  # <- flip
        v_mu    = self.v.predict_under_log(mu_torch, t_torch, requires_grad=True)
        g       = torch.autograd.grad(v_mu.sum(), mu_torch)[0].detach().cpu().numpy()
        if not np.all(np.isfinite(g)):
            print("Warning: non-finite gradient in linearized guidance; setting to zero.")
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Completing the square corresponds to just adding var * g to the mean ---
        mean = mu + var * g                                           # N( μ + σ^2 ∇v_t(μ), σ^2 I )
        return mean + std * np.random.normal(size=mean.shape)


    # def single_step(self, t, x_tplus1):
    #     """
    #     One reverse step using a linearized tilt of v_t(·) around μ_{t|t+1}(x_{t+1}).
    #     """
    #     # Base reverse Gaussian: mean μ, var σ^2 I
    #     mu   = self.tpost.tweedie_posterior_mean_single_step(t, x_tplus1)  # (S,d)
    #     var  = (1.0 - self.alpha_t[t]) / self.alpha_t[t]                   # σ_t^2
    #     std  = np.sqrt(var)

    #     # Gradient g = ∇_x v_t(x) evaluated at x=μ
    #     mu_torch = torch.tensor(mu, dtype=torch.float32, requires_grad=True)
    #     # fo_guidance.py
    #     t_torch = torch.full((mu.shape[0],), float(len(self.alpha_t) - 1 - t), dtype=torch.float32)
    #     v_mu     = self.v.predict_under_log(mu_torch, t_torch, requires_grad = True)              # shape (S,)
    #     g        = torch.autograd.grad(v_mu.sum(), mu_torch)[0].detach().cpu().numpy()  # (S,d)

    #     # Complete the square: N( μ + σ^2 g, σ^2 I )
    #     mean = mu + var * g

    #     return mean + std * np.random.normal(size=mean.shape)
