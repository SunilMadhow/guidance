# chernoff_guidance.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from guided_models import Diffusion  # uses single_step_iid_copies etc.


# ----------------------------
# Baseline network: predicts log u_{t+1}(y_{t+1})
# Input follows your value nets' pattern: concat [x, time_frac]
# ----------------------------
class LogUBaselineNet(nn.Module):
    def __init__(self, d, T, hidden=(64, 64), device=None, lr=1e-3):
        super().__init__()
        self.d = d
        self.T = T
        self.device = device or (torch.device("cuda") if torch.cuda.is_available()
                                 else torch.device("cpu"))
        in_dim = d + 1
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers).to(self.device)
        self.opt = optim.SGD(self.net.parameters(), lr=lr)

    def forward(self, x, t_scalar):
        """
        x: (N, d) float tensor
        t_scalar: int time index OR (N,) normalized to [0,1]
        returns: (N,) float tensor = log u_{t+1}(x)
        """
        if isinstance(t_scalar, int):
            t_feat = float(t_scalar) / (self.T - 1)
            t = torch.full((x.shape[0], 1), t_feat, dtype=x.dtype, device=x.device)
        else:
            t = t_scalar.view(-1, 1).to(x.device).to(x.dtype)
        inp = torch.cat([x, t], dim=-1)
        out = self.net(inp).squeeze(-1)
        return out  # log u


def visualize_baseline_2d(baseline: LogUBaselineNet, t_scalar: int,
                          grid_size: int = 100, value_range=(-3, 3)):
    """
    Visualizes log u_{t+1}(y_{t+1}) over a 2D grid for d=2.
    """
    import matplotlib.pyplot as plt
    if baseline.d != 2:
        raise ValueError("Visualization only supported for d=2.")
    x = np.linspace(value_range[0], value_range[1], grid_size)
    y = np.linspace(value_range[0], value_range[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (grid_size^2, 2)
    with torch.no_grad():
        grid_tensor = torch.tensor(grid, dtype=torch.float32, device=baseline.device)
        logu = baseline(grid_tensor, t_scalar).cpu().numpy().reshape(grid_size, grid_size)
    plt.figure(figsize=(6, 5))
    plt.imshow(logu, extent=(value_range[0], value_range[1], value_range[0], value_range[1]),
               origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='log u')
    plt.title(f"log u_{{t+1}}(y_{{t+1}}), t={t_scalar}")
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.show()


# ----------------------------
# Chernoff-guided diffusion with SHARED time-conditioned baseline
# ----------------------------
class ChernoffGuidance(Diffusion):
    """
    Single time-conditioned baseline net:
        baseline(x, t_scalar) -> log u_{t+1}(x)
    Per-step scalars: lambdas[t], taus[t].
    """

    def __init__(self,
                 dist0,
                 alpha_t,
                 value_fn,                # returns log-values v_t(x)
                 delta=1e-3,
                 S=10_000,
                 hidden=(64, 64),
                 lr=1e-3,
                 lam_lr = 1e-3,
                 iters=200,
                 erm_error=0.0,          # SLT/ERM uniform slack added to tau
                 verbose=False,
                 fix_lambda=False,       # if True, do NOT train lambda
                 lambda_value=0.5,       # used when fix_lambda=True
                 reverse_value_time=False):  # if True, feed T-1-t to value_fn
        super().__init__(dist0, alpha_t, S)
        self.value_fn = value_fn
        self.delta = float(delta)
        self.T = len(alpha_t)
        self.hidden = hidden
        self.lr = lr
        self.iters = int(iters)
        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else torch.device("cpu"))
        self.verbose = verbose

        # Manual lambda control
        self.fix_lambda = bool(fix_lambda)
        self.lambda_value = float(lambda_value)
        self.lam_lr = lam_lr

        # Whether the value_fn expects reversed time index
        self.reverse_value_time = bool(reverse_value_time)

        # Shared baseline net (time-conditioned)
        self.baseline = LogUBaselineNet(d=self.d, T=self.T,
                                        hidden=self.hidden, lr=self.lr).to(self.device)

        # Per-step Chernoff params
        self.lambdas = np.zeros(self.T - 1, dtype=np.float64)
        self.taus = np.zeros(self.T - 1, dtype=np.float64)

        # ERM slack added directly to tau
        self.erm_error = float(erm_error)

        # Proposal accounting
        self.num_steps = 0

    def __tostr__(self):
        return (f"ChernoffGuidance(S={self.S}, T={self.T}, d={self.d}, "
                f"delta={self.delta}, iters={self.iters}, erm_error={self.erm_error}, "
                f"taus={self.taus}, lambdas={self.lambdas}), num_steps={self.num_steps}")

    # ---------------- helper: value eval (API expects log-values) ----------------
    def _value_eval_numpy(self, t, X):
        """
        Evaluate v_t(X) robustly with time mapping:
          - call value_fn(t_mapped, X) where t_mapped may be scalar or array
          - accept torch or numpy outputs; return shape (n,) numpy
        """
        X = np.asarray(X)
        n = X.shape[0]
        try:
            out = self.value_fn(t, X)
        except TypeError:
            out = self.value_fn(np.full((n,), t), X)
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        out = np.asarray(out).reshape(n)
        if self.verbose:
            kind = "scalar" if np.isscalar(t) or (np.ndim(t) == 0) else "array"
            print(f"[value_eval] t({kind})={t} -> t_mapped={t} X.shape={X.shape} -> v.shape={out.shape}")
        return out

    # ---------------- ONE proposal per prefix (training) ----------------
    @torch.no_grad()
    def _sample_joint_pairs(self, t, Ytp1_batch):
        """
        Returns:
          Ytp1: (B, d) np array
          Xt:   (B, d) np array with ONE proposal per prefix
        """
        Ytp1_batch = np.asarray(Ytp1_batch)
        B, d = Ytp1_batch.shape
        Xprop = self.single_step_iid_copies(t, Ytp1_batch, size=1)  # (1, B, d)
        Xt = Xprop.reshape(B, d)
        if self.verbose:
            print(f"[pairs] t={t} Ytp1.shape={Ytp1_batch.shape} "
                  f"Xprop.shape={Xprop.shape} Xt.shape={Xt.shape}")
        return Ytp1_batch, Xt

    # ---------------- compute tau with erm slack ----------------
    def _compute_tau_with_slack(self, s_tensor, lam_float):
        lam = torch.tensor(lam_float, dtype=torch.float32, device=self.device)
        m = torch.max(lam * s_tensor)
        lme = m + torch.log(torch.mean(torch.exp(lam * s_tensor - m)))
        tau = (torch.log(torch.tensor(1.0 / self.delta, device=self.device)) + lme) / lam
        tau_val = float(tau.item() + self.erm_error)  # add uniform concentration slack
        if self.verbose:
            print(f"[tau] lam={lam_float:.4f} lme={float(lme.item()):.6f} "
                  f"base_tau={float(tau.item()-self.erm_error):.6f} "
                  f"erm={self.erm_error:.6f} tau={tau_val:.6f}")
        return tau_val

    # ---------------- train baseline at step t (for b_{t+1}) ----------------
    def train_baseline_step(self, t, Ytp1, batch_size=1024, lam_init=0.5):
        """
        Trains the SHARED baseline on (y_{t+1}, x_t) with ONE x_t per prefix.
        Updates self.lambdas[t], self.taus[t]. Reuses self.baseline parameters.
        """
        print(f"Training baseline and lambda at step t={t} ...")
        Ytp1 = np.asarray(Ytp1)
        N, d = Ytp1.shape
        if self.verbose:
            print(f"[train_b] t={t} Ytp1.shape={Ytp1.shape} N={N} d={d} "
                  f"fix_lambda={self.fix_lambda} lambda_value={self.lambda_value}")

        # --- Lambda setup (either learn or fix) ---
        if self.fix_lambda:
            lam_param = None
            lam_opt = None
            lam_const = torch.tensor(self.lambda_value, dtype=torch.float32, device=self.device)
        else:
            lam_param = torch.nn.Parameter(
                torch.tensor(np.log(np.expm1(lam_init)), dtype=torch.float32, device=self.device)
            )
            lam_opt = optim.SGD([lam_param], lr=self.lam_lr)

        for it in tqdm(range(self.iters), desc=f"Chernoff b_(t+1) @ t={t}"):
            B = min(batch_size, N)
            idx = np.random.choice(N, size=B, replace=False)
            Ytp1_b = Ytp1[idx]  # (B, d)

            # ONE proposal per prefix
            Ytp1_b, Xt_b = self._sample_joint_pairs(t, Ytp1_b)  # (B, d), (B, d)

            # v_t on proposals (log-values)
            t_arr = np.full(Xt_b.shape[0], self.T - t - 1, dtype=int)
            vt_np = self._value_eval_numpy(t_arr, Xt_b)         # (B,)
            vt = torch.tensor(vt_np, dtype=torch.float32, device=self.device)

            # log u_{t+1} on prefixes with time index t+1 (shared baseline)
            ypref = torch.tensor(Ytp1_b, dtype=torch.float32, device=self.device)
            logu = self.baseline(ypref, t+1)                    # (B,)

            # s_t = v_t(x_t) - log u_{t+1}(y_{t+1})
            s = vt - logu                                        # (B,)

            # pick lambda
            if self.fix_lambda:
                lam = lam_const
            else:
                lam = torch.nn.functional.softplus(lam_param) + 1e-6

            # J_exact + erm_error additive slack
            # term1 = (1/λ) log mean exp(λ s)
            m1 = torch.max(lam * s)
            term1 = (m1 + torch.log(torch.mean(torch.exp(lam * s - m1)))) / lam
            # term2 = log mean exp(-s)
            m2 = torch.max(-s)
            term2 = m2 + torch.log(torch.mean(torch.exp(-s - m2)))
            # term3 = (1/λ) log(1/δ)
            term3 = torch.log(torch.tensor(1.0 / self.delta, device=self.device)) / lam
            # term4 = erm slack (added to tau) enters additively in loss gauge
            term4 = torch.tensor(self.erm_error, dtype=torch.float32, device=self.device)

            loss = term1 + term2 + term3 + term4

            if self.verbose and (it % max(1, self.iters // 5) == 0):
                print(f"[iter:{it}] B={B} vt.shape={vt.shape} logu.shape={logu.shape} s.shape={s.shape}")
                print(f"          lam={float(lam.item()):.6f} "
                      f"t1={float(term1.item()):.6f} t2={float(term2.item()):.6f} "
                      f"t3={float(term3.item()):.6f} erm={self.erm_error:.6f} "
                      f"loss={float(loss.item()):.6f}")

            self.baseline.opt.zero_grad()
            if lam_opt is not None:
                lam_opt.zero_grad()
            loss.backward()
            self.baseline.opt.step()
            if lam_opt is not None:
                lam_opt.step()

        # finalize λ_t and compute τ_t using a shallow pass (one proposal per prefix)
        if self.fix_lambda:
            lambda_t = float(self.lambda_value)
        else:
            lambda_t = float(torch.nn.functional.softplus(lam_param).item() + 1e-6)

        with torch.no_grad():
            _, Xt_all = self._sample_joint_pairs(t, Ytp1)
            t_arr = np.full(Xt_all.shape[0], self.T - t - 1, dtype=int)
            vt_np = self._value_eval_numpy(t_arr, Xt_all)
            vt = torch.tensor(vt_np, dtype=torch.float32, device=self.device)
            logu_all = self.baseline(torch.tensor(Ytp1, dtype=torch.float32, device=self.device), t+1)
            s_all = vt - logu_all
            # numerical sanity
            if not torch.isfinite(s_all).all():
                raise FloatingPointError("Non-finite s_all detected in tau computation.")
            tau_t = self._compute_tau_with_slack(s_all, lambda_t)

        self.lambdas[t] = lambda_t
        self.taus[t] = tau_t
        return lambda_t, tau_t

    # ---------------- guided sampling with acceptance ----------------
    def single_step_guide(self, t, Ytp1, batch_size=16):
        """
        Accept if u < exp( tau_t + logu_{t+1}(y_{t+1}) - v_t(x_t) ).
        Uses the SHARED baseline net and stored τ_t.
        """
        Ytp1 = np.asarray(Ytp1)
        K, d = Ytp1.shape
        out = np.empty((K, d))
        done = np.zeros(K, dtype=bool)

        tau = self.taus[t]
        if self.verbose:
            print(f"[guide] t={t} K={K} d={d} tau={tau:.6f}")

        while not done.all():
            active = np.where(~done)[0]
            Ka = active.size
            Xprop = super().single_step_iid_copies(t, Ytp1[active], size=batch_size)  # (B, Ka, d)
            Xflat = Xprop.reshape(batch_size * Ka, d)

            # v_t on proposals
            t_arr = np.full(Xflat.shape[0], self.T - t - 1, dtype=int)
            v_vals = self._value_eval_numpy(t_arr, Xflat).reshape(batch_size, Ka)  # (B, Ka)

            # log u on prefixes (broadcast over B)
            with torch.no_grad():
                ypref = torch.tensor(Ytp1[active], dtype=torch.float32, device=self.device)
                logu_vals = self.baseline(ypref, t+1).detach().cpu().numpy()      # (Ka,)

            if self.verbose:
                print(f"[guide] active={Ka} Xprop.shape={Xprop.shape} "
                      f"v_vals.shape={v_vals.shape} logu_vals.shape={logu_vals.shape}")

            # acceptance exponent: tau + logu(y_{t+1}) - v_t(x_t)
            accept_log_exponent = -1*(tau + logu_vals[None, :] - v_vals)              # (B, Ka)
            if self.verbose:
                print(f"[guide] accept_log_exponent.shape={accept_log_exponent.shape} "
                      f"mean={accept_log_exponent.mean():.6f} "
                      f"min={accept_log_exponent.min():.6f} "
                      f"max={accept_log_exponent.max():.6f}")

            # clip at 1: exp(min(0, ·))
            R = np.exp(accept_log_exponent)
            print(R)
            U = np.random.rand(batch_size, Ka)
            accept = (U < R)

            # proposal accounting
            for k in range(Ka):
                idx = np.where(accept[:, k])[0]
                self.num_steps += (idx[0] + 1) if idx.size > 0 else batch_size

            got_any = accept.any(axis=0)
            if got_any.any():
                first_idx = np.argmax(accept, axis=0)
                sel = active[got_any]
                picks = first_idx[got_any]
                out[sel] = Xprop[picks, got_any, :]
                done[sel] = True

        return out

    # ---------------- overall backward train+sample ----------------
    def train_baselines_and_sample(self, batch_size=1024, lam_init = 0.5):
        """
        1) Initialize y_T ~ N(0, I).
        2) For t = T-2 ... 0:
           • Train SHARED baseline + (learned OR fixed) λ_t on {(y_{t+1}^i, x_t^i)}.
           • Compute τ_t with ERM slack.
           • Sample y_t via rejection using (baseline, τ_t).
        Returns X in chronological order (S, T, d).
        """
        T = len(self.alpha_t)
        X = np.zeros((self.S, T, self.d))
        X[:, -1, :] = np.random.normal(size=(self.S, self.d))

        for t in tqdm(range(T - 2, -1, -1), desc="Chernoff backward"):
            Ytp1 = X[:, t + 1, :]
            if self.verbose:
                print(f"[loop] t={t} Ytp1.shape={Ytp1.shape}")

            lam_t, tau_t = self.train_baseline_step(t=t, Ytp1=Ytp1, batch_size=batch_size, lam_init=lam_init)
            if self.verbose:
                print(f"[loop] t={t} lambda_t={lam_t:.6f} tau_t={tau_t:.6f}")

            X[:, t, :] = self.single_step_guide(t, X[:, t + 1, :])

        # flip to chronological (x_0 ... x_T)
        return np.flip(X, axis=1)

    # ---------------- inference-only sampler ----------------
    def calc_samples(self, S=5000, YT=None, batch_size=16, reset_counter=True):
        """
        Inference-only sampling with the SHARED baseline net and stored taus.
        """
        # ---- Sanity checks ----
        if self.baseline is None:
            raise RuntimeError("Baseline net is None. Train or load weights before inference.")
        if np.any([np.isnan(t) or np.isinf(t) for t in self.taus]):
            raise RuntimeError("Invalid tau values. Ensure taus[] are computed before inference.")

        # ---- Initialization ----
        if YT is not None:
            YT = np.asarray(YT)
            if YT.ndim != 2 or YT.shape[1] != self.d:
                raise ValueError(f"YT must have shape (S, d); got {YT.shape}.")
            S_eff = YT.shape[0]
        else:
            S_eff = S if S is not None else self.S
            YT = np.random.normal(size=(S_eff, self.d))

        if reset_counter:
            self.num_steps = 0

        T = len(self.alpha_t)
        X = np.zeros((S_eff, T, self.d), dtype=float)
        X[:, -1, :] = YT

        if self.verbose:
            print(f"[sample_only] S={S_eff} T={T} d={self.d}")
            print(f"[sample_only] YT.shape={YT.shape}")

        for t in tqdm(range(T - 2, -1, -1), desc="Chernoff inference", leave=True):
            Ytp1 = X[:, t + 1, :]
            if self.verbose:
                print(f"[sample_only] t={t} Ytp1.shape={Ytp1.shape} tau={self.taus[t]:.6f}")
            X[:, t, :] = self.single_step_guide(t, Ytp1, batch_size=batch_size)

        X_chrono = np.flip(X, axis=1)
        if self.verbose:
            print(f"[sample_only] done. X_chrono.shape={X_chrono.shape} num_steps={self.num_steps}")
        return X_chrono



# # chernoff_guidance.py

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm

# from guided_models import Diffusion  # uses single_step_iid_copies etc.

# # ----------------------------
# # Baseline network: predicts log u_{t+1}(y_{t+1})
# # (same I/O pattern as your value nets: concat [x, time_frac])
# # ----------------------------
# class LogUBaselineNet(nn.Module):
#     def __init__(self, d, T, hidden=(64, 64), device=None, lr=1e-3):
#         super().__init__()
#         self.d = d
#         self.T = T
#         self.device = device or (torch.device("cuda") if torch.cuda.is_available()
#                                  else torch.device("cpu"))
#         layers = []
#         in_dim = d + 1
#         for h in hidden:
#             layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
#             in_dim = h
#         layers += [nn.Linear(in_dim, 1)]
#         self.net = nn.Sequential(*layers).to(self.device)
#         self.opt = optim.SGD(self.net.parameters(), lr=lr)
        

#     def forward(self, x, t_scalar):
#         """
#         x: (N, d) float tensor
#         t_scalar: int time index or (N,) normalized to [0,1]
#         returns: (N,) float tensor = log u_{t+1}(x)
#         """
#         if isinstance(t_scalar, int):
#             t_feat = float(t_scalar) / (self.T - 1)
#             t = torch.full((x.shape[0], 1), t_feat, dtype=x.dtype, device=x.device)
#         else:
#             t = t_scalar.view(-1, 1).to(x.device).to(x.dtype)
#         inp = torch.cat([x, t], dim=-1)
#         out = self.net(inp).squeeze(-1)
#         return out  # log u
#     def visualize_baseline_2d(baseline, t_scalar, grid_size=100, value_range=(-3, 3)):
#         """
#         Visualizes log u_{t+1}(y_{t+1}) over a 2D grid for d=2.
#         Args:
#             baseline: LogUBaselineNet instance
#             t_scalar: time index (int)
#             grid_size: number of points per axis
#             value_range: tuple (min, max) for each axis
#         """
#         import matplotlib.pyplot as plt

#         if baseline.d != 2:
#             raise ValueError("Visualization only supported for d=2.")

#         x = np.linspace(value_range[0], value_range[1], grid_size)
#         y = np.linspace(value_range[0], value_range[1], grid_size)
#         xx, yy = np.meshgrid(x, y)
#         grid = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (grid_size^2, 2)

#         with torch.no_grad():
#             grid_tensor = torch.tensor(grid, dtype=torch.float32, device=baseline.device)
#             logu = baseline(grid_tensor, t_scalar).cpu().numpy().reshape(grid_size, grid_size)

#         plt.figure(figsize=(6, 5))
#         plt.imshow(logu, extent=(value_range[0], value_range[1], value_range[0], value_range[1]),
#                    origin='lower', aspect='auto', cmap='viridis')
#         plt.colorbar(label='log u')
#         plt.title(f"log u_{{t+1}}(y_{{t+1}}), t={t_scalar}")
#         plt.xlabel("y1")
#         plt.ylabel("y2")
#         plt.show()
    
    
# class ChernoffGuidance(Diffusion):
#     """
#     Single time-conditioned baseline net:
#         baseline(x, t_scalar) -> log u_{t+1}(x)
#     Per-step scalars: lambdas[t], taus[t].
#     """
#     def __init__(self,
#                  dist0,
#                  alpha_t,
#                  value_fn,                # returns log-values v_t(x)
#                  delta=1e-3,
#                  S=10_000,
#                  hidden=(64, 64),
#                  lr=1e-3,
#                  iters=200,
#                  erm_error=0.0,          # SLT/ERM uniform slack added to tau
#                  verbose=False):
#         super().__init__(dist0, alpha_t, S)
#         self.value_fn = value_fn
#         self.delta = float(delta)
#         self.T = len(alpha_t)
#         self.hidden = hidden
#         self.lr = lr
#         self.iters = int(iters)
#         self.device = (torch.device("cuda") if torch.cuda.is_available()
#                        else torch.device("cpu"))
#         self.verbose = verbose

#         # Single baseline net shared across all steps (time-conditioned)
#         self.baseline = LogUBaselineNet(d=self.d, T=self.T,
#                                         hidden=self.hidden, lr=self.lr).to(self.device)

#         # Per-step Chernoff params
#         self.lambdas = np.zeros(self.T - 1, dtype=np.float64)
#         self.taus    = np.zeros(self.T - 1, dtype=np.float64)

#         # ERM slack added directly to tau
#         self.erm_error = float(erm_error)
#         self.num_steps = 0

#     # ---------------- helper: value eval (API expects log-values) ----------------
#     def _value_eval_numpy(self, t, X):
#         X = np.asarray(X)
#         n = X.shape[0]
#         try:
#             out = self.value_fn(t, X)
#         except TypeError:
#             out = self.value_fn(np.full((n,), t), X)
#         if isinstance(out, torch.Tensor):
#             out = out.detach().cpu().numpy()
#         out = np.asarray(out).reshape(n)
#         if self.verbose:
#             print(f"[value_eval] t={t if np.isscalar(t) else 'arr'} X.shape={X.shape} -> v.shape={out.shape}")
#         return out

#     # ---------------- ONE proposal per prefix (training) ----------------
#     @torch.no_grad()
#     def _sample_joint_pairs(self, t, Ytp1_batch):
#         Ytp1_batch = np.asarray(Ytp1_batch)
#         B, d = Ytp1_batch.shape
#         Xprop = self.single_step_iid_copies(t, Ytp1_batch, size=1)  # (1,B,d)
#         Xt = Xprop.reshape(B, d)
#         if self.verbose:
#             print(f"[pairs] t={t} Ytp1.shape={Ytp1_batch.shape} Xprop.shape={Xprop.shape} Xt.shape={Xt.shape}")
#         return Ytp1_batch, Xt

#     # ---------------- compute tau with erm slack ----------------
#     def _compute_tau_with_slack(self, s_tensor, lam_float):
#         lam = torch.tensor(lam_float, dtype=torch.float32, device=self.device)
#         m = torch.max(lam * s_tensor)
#         lme = m + torch.log(torch.mean(torch.exp(lam * s_tensor - m)))
#         tau = (torch.log(torch.tensor(1.0 / self.delta, device=self.device)) + lme) / lam
#         tau_val = float(tau.item() + self.erm_error)  # add uniform concentration slack
#         if self.verbose:
#             print(f"[tau] lam={lam_float:.4f} lme={float(lme.item()):.6f} "
#                   f"base_tau={float(tau.item()-self.erm_error):.6f} erm={self.erm_error:.6f} tau={tau_val:.6f}")
#         return tau_val

#     # ---------------- train baseline at step t (for b_{t+1}) ----------------
#     def train_baseline_step(self, t, Ytp1, batch_size=1024, lam_init=0.5):
#         """
#         Trains the SHARED baseline on (y_{t+1}, x_t) with ONE x_t per prefix.
#         Updates self.lambdas[t], self.taus[t]. Reuses self.baseline parameters.
#         """
#         Ytp1 = np.asarray(Ytp1)
#         N, d = Ytp1.shape
#         if self.verbose:
#             print(f"[train_b] t={t} Ytp1.shape={Ytp1.shape} N={N} d={d}")

#         # Learn λ_t > 0 for THIS step (scalar param)
#         lam_param = torch.nn.Parameter(
#             torch.tensor(np.log(np.expm1(lam_init)), dtype=torch.float32, device=self.device)
#         )
#         lam_opt = optim.SGD([lam_param], lr=self.lr)

#         for it in tqdm(range(self.iters), desc=f"Chernoff b_(t+1) @ t={t}"):
#             B = min(batch_size, N)
#             idx = np.random.choice(N, size=B, replace=False)
#             Ytp1_b = Ytp1[idx]  # (B,d)

#             # ONE proposal per prefix
#             Ytp1_b, Xt_b = self._sample_joint_pairs(t, Ytp1_b)  # (B,d), (B,d)

#             # v_t on proposals (log-values)
#             t_arr = (self.T - 1 - t) * np.ones(Xt_b.shape[0])
#             vt_np = self._value_eval_numpy(t_arr, Xt_b)         # (B,)
#             vt = torch.tensor(vt_np, dtype=torch.float32, device=self.device)

#             # log u_{t+1} on prefixes with time index t+1 (shared baseline)
#             ypref = torch.tensor(Ytp1_b, dtype=torch.float32, device=self.device)
#             logu  = self.baseline(ypref, t+1)                   # (B,)

#             # s_t = v_t(x_t) - log u_{t+1}(y_{t+1})
#             s = vt - logu

#             lam = torch.nn.functional.softplus(lam_param) + 1e-6

#             # J_exact + erm_error additive slack
#             m1 = torch.max(lam * s)
#             term1 = (m1 + torch.log(torch.mean(torch.exp(lam * s - m1)))) / lam
#             m2 = torch.max(-s)
#             term2 = m2 + torch.log(torch.mean(torch.exp(-s - m2)))
#             term3 = torch.log(torch.tensor(1.0 / self.delta, device=self.device)) / lam
#             term4 = torch.tensor(self.erm_error, dtype=torch.float32, device=self.device)  # ADDITIVE

#             loss = term1 + term2 + term3 + term4

#             if self.verbose and (it % max(1, self.iters // 5) == 0):
#                 print(f"[iter:{it}] B={B} vt.shape={vt.shape} logu.shape={logu.shape} s.shape={s.shape}")
#                 print(f"          lam={float(lam.item()):.6f} t1={float(term1.item()):.6f} "
#                       f"t2={float(term2.item()):.6f} t3={float(term3.item()):.6f} "
#                       f"erm={self.erm_error:.6f} loss={float(loss.item()):.6f}")

#             self.baseline.opt.zero_grad()
#             lam_opt.zero_grad()
#             loss.backward()
#             self.baseline.opt.step()
#             lam_opt.step()

#         # finalize λ_t and compute τ_t on a shallow pass (one proposal per prefix)
#         lambda_t = float(torch.nn.functional.softplus(lam_param).item() + 1e-6)
#         with torch.no_grad():
#             _, Xt_all = self._sample_joint_pairs(t, Ytp1)
#             t_arr = (self.T - 1 - t) * np.ones(Xt_all.shape[0])
#             vt_np = self._value_eval_numpy(t_arr, Xt_all)
#             vt = torch.tensor(vt_np, dtype=torch.float32, device=self.device)
#             logu_all = self.baseline(torch.tensor(Ytp1, dtype=torch.float32, device=self.device), t+1)
#             s_all = vt - logu_all
#             tau_t = self._compute_tau_with_slack(s_all, lambda_t)

#         self.lambdas[t] = lambda_t
#         self.taus[t]    = tau_t
#         return lambda_t, tau_t

#     # ---------------- guided sampling with acceptance ----------------
#     def single_step_guide(self, t, Ytp1, batch_size=16):
#         """
#         Accept if u < exp( tau_t + logu_{t+1}(y_{t+1}) - v_t(x_t) ).
#         Uses the SHARED baseline net.
#         """
#         Ytp1 = np.asarray(Ytp1)
#         K, d = Ytp1.shape
#         out = np.empty((K, d))
#         done = np.zeros(K, dtype=bool)

#         tau = self.taus[t]
#         if self.verbose:
#             print(f"[guide] t={t} K={K} d={d} tau={tau:.6f}")

#         while not done.all():
#             active = np.where(~done)[0]
#             Ka = active.size
#             Xprop = super().single_step_iid_copies(t, Ytp1[active], size=batch_size)  # (B,Ka,d)
#             Xflat = Xprop.reshape(batch_size * Ka, d)

#             # v_t on proposals
#             t_arr = (self.T - 1 - t) * np.ones(Xflat.shape[0])
#             v_vals = self._value_eval_numpy(t_arr, Xflat).reshape(batch_size, Ka)  # (B,Ka)

#             # log u on prefixes (broadcast over B)
#             with torch.no_grad():
#                 ypref = torch.tensor(Ytp1[active], dtype=torch.float32, device=self.device)
#                 logu_vals = self.baseline(ypref, t+1).detach().cpu().numpy()     # (Ka,)

#             if self.verbose:
#                 print(f"[guide] active={Ka} Xprop.shape={Xprop.shape} v_vals.shape={v_vals.shape} logu_vals.shape={logu_vals.shape}")

#             # acceptance exponent: tau + logu(y_{t+1}) - v_t(x_t)
#             accept_log_exponent = tau + logu_vals[None, :] - v_vals              # (B,Ka)
#             if self.verbose:
#                 print(f"[guide] accept_log_exponent.shape={accept_log_exponent.shape} "
#                       f"mean={accept_log_exponent.mean():.6f} min={accept_log_exponent.min():.6f} max={accept_log_exponent.max():.6f}")

#             # clip at 1: exp(min(0, ·))
#             R = np.exp(np.minimum(0.0, accept_log_exponent))
#             U = np.random.rand(batch_size, Ka)
#             accept = (U < R)

#             # proposal accounting
#             for k in range(Ka):
#                 idx = np.where(accept[:, k])[0]
#                 self.num_steps += (idx[0] + 1) if idx.size > 0 else batch_size

#             got_any = accept.any(axis=0)
#             if got_any.any():
#                 first_idx = np.argmax(accept, axis=0)
#                 sel = active[got_any]
#                 picks = first_idx[got_any]
#                 out[sel] = Xprop[picks, got_any, :]
#                 done[sel] = True

#         return out

#     # ---------------- overall backward train+sample ----------------
#     def train_baselines_and_sample(self, batch_size=1024):
#         """
#         1) Initialize y_T ~ N(0, I).
#         2) For t = T-2 ... 0:
#            • Train SHARED baseline + λ_t on {(y_{t+1}^i, x_t^i)} (one x_t per prefix).
#            • Compute τ_t with ERM slack.
#            • Sample y_t via rejection using (baseline, τ_t).
#         Returns X in chronological order (S, T, d).
#         """
#         T = len(self.alpha_t)
#         X = np.zeros((self.S, T, self.d))
#         X[:, -1, :] = np.random.normal(size=(self.S, self.d))

#         for t in tqdm(range(T - 2, -1, -1), desc="Chernoff backward"):
#             Ytp1 = X[:, t + 1, :]
#             if self.verbose:
#                 print(f"[loop] t={t} Ytp1.shape={Ytp1.shape}")

#             lam_t, tau_t = self.train_baseline_step(t=t, Ytp1=Ytp1, batch_size=batch_size)
#             if self.verbose:
#                 print(f"[loop] t={t} lambda_t={lam_t:.6f} tau_t={tau_t:.6f}")

#             X[:, t, :] = self.single_step_guide(t, X[:, t + 1, :])

#         # flip to chronological (x_0 ... x_T)
#         return np.flip(X, axis=1)

#     def calc_samples(self, S=5000, YT=None, batch_size=16, reset_counter=True):
#         """
#         Inference-only sampling with the SHARED baseline net and stored taus.
#         """
#         # ---- Sanity checks ----
#         if self.baseline is None:
#             raise RuntimeError("Baseline net is None. Train or load weights before inference.")
#         if np.any([np.isnan(t) or np.isinf(t) for t in self.taus]):
#             raise RuntimeError("Invalid tau values. Ensure taus[] are computed before inference.")

#         # ---- Initialization ----
#         if YT is not None:
#             YT = np.asarray(YT)
#             if YT.ndim != 2 or YT.shape[1] != self.d:
#                 raise ValueError(f"YT must have shape (S, d); got {YT.shape}.")
#             S_eff = YT.shape[0]
#         else:
#             S_eff = S if S is not None else self.S
#             YT = np.random.normal(size=(S_eff, self.d))

#         if reset_counter:
#             self.num_steps = 0

#         T = len(self.alpha_t)
#         X = np.zeros((S_eff, T, self.d), dtype=float)
#         X[:, -1, :] = YT

#         if self.verbose:
#             print(f"[sample_only] S={S_eff} T={T} d={self.d}")
#             print(f"[sample_only] YT.shape={YT.shape}")

#         for t in tqdm(range(T - 2, -1, -1), desc="Chernoff inference", leave=True):
#             Ytp1 = X[:, t + 1, :]
#             if self.verbose:
#                 print(f"[sample_only] t={t} Ytp1.shape={Ytp1.shape} tau={self.taus[t]:.6f}")
#             X[:, t, :] = self.single_step_guide(t, Ytp1, batch_size=batch_size)

#         X_chrono = np.flip(X, axis=1)
#         if self.verbose:
#             print(f"[sample_only] done. X_chrono.shape={X_chrono.shape} num_steps={self.num_steps}")
#         return X_chrono



# # class ChernoffGuidance(Diffusion):
# #     """
# #     Trains per-step baselines b_{t+1}(y_{t+1}) as log u_{t+1}(y_{t+1}),
# #     using the Chernoff objective with an ERM uniform-concentration slack.
# #     Then samples with acceptance:
# #         alpha = min{1, exp( tau_t + log u_{t+1}(y_{t+1}) - v_t(x_t) )}
# #     """
# #     def __init__(self,
# #                  dist0,
# #                  alpha_t,
# #                  value_fn,                # returns log-values v_t(x)
# #                  delta=1e-3,
# #                  S=10_000,
# #                  hidden=(64, 64),
# #                  lr=1e-3,
# #                  iters=200,
# #                  erm_error=0.0,          # SLT/ERM uniform slack added to tau
# #                  verbose=False):
# #         super().__init__(dist0, alpha_t, S)
# #         self.value_fn = value_fn
# #         self.delta = float(delta)
# #         self.T = len(alpha_t)
# #         self.hidden = hidden
# #         self.lr = lr
# #         self.iters = int(iters)
# #         self.device = (torch.device("cuda") if torch.cuda.is_available()
# #                        else torch.device("cpu"))
# #         self.verbose = verbose

# #         self.baselines = [None for _ in range(self.T - 1)]  # log-u nets for t+1
# #         self.lambdas = np.zeros(self.T - 1, dtype=np.float64)
# #         self.taus = np.zeros(self.T - 1, dtype=np.float64)

# #         # ERM slack that bumps tau_t ← tau_t + erm_error
# #         self.erm_error = float(erm_error)
# #         self.num_steps = 0

# #     # ---------------- helper: value eval (API expects log-values) ----------------
# #     def _value_eval_numpy(self, t, X):
# #         """
# #         Evaluate v_t(X) robustly:
# #           - try value_fn(t, X)
# #           - if needed, try value_fn(np.full(n, t), X)
# #           - accept torch or numpy outputs; return (n,) numpy
# #         """
# #         X = np.asarray(X)
# #         n = X.shape[0]
# #         try:
# #             out = self.value_fn(t, X)
# #         except TypeError:
# #             out = self.value_fn(np.full((n,), t), X)
# #         # to numpy, squeeze
# #         try:
# #             import torch
# #             if isinstance(out, torch.Tensor):
# #                 out = out.detach().cpu().numpy()
# #         except ImportError:
# #             pass
# #         out = np.asarray(out).reshape(n)
# #         return out

# #     # ---------------- ONE proposal per prefix (training) ----------------
# #     @torch.no_grad()
# #     def _sample_joint_pairs(self, t, Ytp1_batch):
# #         """
# #         Returns:
# #           Ytp1: (B, d) np array
# #           Xt:   (B, d) np array with ONE proposal per prefix
# #         """
# #         Ytp1_batch = np.asarray(Ytp1_batch)
# #         B, d = Ytp1_batch.shape
# #         Xprop = self.single_step_iid_copies(t, Ytp1_batch, size = 1)  # (1,B,d)
# #         print("Xprop shape:", Xprop.shape)
# #         Xt = Xprop.reshape(B, d)
# #         if self.verbose:
# #             print(f"[pairs] t={t} Ytp1_batch.shape={Ytp1_batch.shape} Xprop.shape={Xprop.shape} Xt.shape={Xt.shape}")
# #         return Ytp1_batch, Xt

# #     # ---------------- compute tau with erm slack ----------------
# #     def _compute_tau_with_slack(self, s_tensor, lam_float):
# #         lam = torch.tensor(lam_float, dtype=torch.float32, device=self.device)
# #         m = torch.max(lam * s_tensor)
# #         lme = m + torch.log(torch.mean(torch.exp(lam * s_tensor - m)))
# #         tau = (torch.log(torch.tensor(1.0 / self.delta, device=self.device)) + lme) / lam
# #         tau_val = float(tau.item() + self.erm_error)  # add uniform concentration slack
# #         if self.verbose:
# #             print(f"[tau] lam={lam_float:.4f} lme={float(lme.item()):.6f} base_tau={float((tau-self.erm_error).item()):.6f} erm={self.erm_error:.6f} tau={tau_val:.6f}")
# #         return tau_val

# #     # ---------------- train baseline at step t (for b_{t+1}) ----------------
# #     def train_baseline_step(self, t, Ytp1, batch_size=1024, lam_init=0.5):
# #         """
# #         Trains b_{t+1}(y_{t+1}) = log u_{t+1}(y_{t+1}) on pairs (y_{t+1}, x_t),
# #         with exactly ONE pre-trained sample per prefix.
# #         """
# #         Ytp1 = np.asarray(Ytp1)
# #         N, d = Ytp1.shape
# #         if self.verbose:
# #             print(f"[train_b] t={t} Ytp1.shape={Ytp1.shape} N={N} d={d}")

# #         bnet = LogUBaselineNet(d=d, T=self.T, hidden=self.hidden, lr=self.lr).to(self.device)

# #         # learn λ_t > 0
# #         lam_param = torch.nn.Parameter(
# #             torch.tensor(np.log(np.expm1(lam_init)), dtype=torch.float32, device=self.device)
# #         )
# #         lam_opt = optim.SGD([lam_param], lr=self.lr)

# #         # iterations (no epochs; each iter uses a fresh minibatch + fresh proposals)
# #         for it in tqdm(range(self.iters), desc=f"Chernoff b_(t+1) @ t={t}"):
# #             B = min(batch_size, N)
# #             idx = np.random.choice(N, size=B, replace=False)
# #             Ytp1_b = Ytp1[idx]                         # (B,d)

# #             # ONE proposal per prefix
# #             Ytp1_b, Xt_b = self._sample_joint_pairs(t, Ytp1_b)  # (B,d), (B,d)
# #             # print("Xt_b shape:", Xt_b.shape)

# #             # v_t on proposals (log-values)
# #             t_arr = (self.T - 1 - t) * np.ones(Xt_b.shape[0])
# #             # print("t_arr shape:", t_arr.shape)
# #             vt_np = self._value_eval_numpy(t_arr, Xt_b)             # (B,)
# #             # print("vt_np shape:", vt_np.shape)
# #             vt = torch.tensor(vt_np, dtype=torch.float32, device=self.device)

# #             # log u_{t+1} on prefixes with time index t+1
# #             ypref = torch.tensor(Ytp1_b, dtype=torch.float32, device=self.device)
# #             vtp1 = bnet(ypref, t+1)                             # (B,)

# #             # s_t = v_t(x_t) - b_{t+1}(y_{t+1}) = v_t - log u_{t+1}
# #             s = vt - vtp1                                       # (B,)

# #             lam = torch.nn.functional.softplus(lam_param) + 1e-6

# #             # J_exact with erm_error added to tau (i.e., add erm_error to loss)
# #             # term1 = (1/λ) log mean exp(λ s)
# #             m1 = torch.max(lam * s)
# #             term1 = (m1 + torch.log(torch.mean(torch.exp(lam * s - m1)))) / lam
# #             # term2 = log mean exp(-s)
# #             m2 = torch.max(-s)
# #             term2 = m2 + torch.log(torch.mean(torch.exp(-s - m2)))
# #             # term3 = (1/λ) log(1/δ)
# #             term3 = torch.log(torch.tensor(1.0 / self.delta, device=self.device)) / lam
# #             # erm term (added to tau) enters additively
# #             term4 = torch.log(torch.tensor(self.erm_error, dtype=torch.float32, device=self.device))

# #             loss = term1 + term2 + term3 + term4

# #             # verbose shapes / scalars
# #             if self.verbose and (it % max(1, self.iters // 5) == 0):
# #                 print(f"[iter:{it}] B={B} vt.shape={vt.shape} logu.shape={logu.shape} s.shape={s.shape}")
# #                 print(f"          lam={float(lam.item()):.6f} term1={float(term1.item()):.6f} term2={float(term2.item()):.6f} term3={float(term3.item()):.6f} erm={self.erm_error:.6f} loss={float(loss.item()):.6f}")

# #             bnet.opt.zero_grad()
# #             lam_opt.zero_grad()
# #             loss.backward()
# #             bnet.opt.step()
# #             lam_opt.step()

# #         # finalize λ and compute τ on a shallow “all-prefix” pass (still one proposal each)
# #         lambda_t = float(torch.nn.functional.softplus(lam_param).item() + 1e-6)
# #         with torch.no_grad():
# #             _, Xt_all = self._sample_joint_pairs(t, Ytp1)
# #             t_arr = (self.T - 1 - t) * np.ones(Xt_all.shape[0])
# #             vt_np = self._value_eval_numpy(t_arr, Xt_all)
# #             vt = torch.tensor(vt_np, dtype=torch.float32, device=self.device)
# #             logu_all = bnet(torch.tensor(Ytp1, dtype=torch.float32, device=self.device), t+1)
# #             s_all = vt - logu_all
# #             tau_t = self._compute_tau_with_slack(s_all, lambda_t)

# #         return bnet, lambda_t, tau_t

# #     # ---------------- guided sampling with acceptance ----------------
# #     def single_step_guide(self, t, Ytp1, batch_size=16):
# #         """
# #         Accept if u < exp( tau_t + logu_{t+1}(y_{t+1}) - v_t(x_t) ).
# #         """
# #         Ytp1 = np.asarray(Ytp1)
# #         K, d = Ytp1.shape
# #         out = np.empty((K, d))
# #         done = np.zeros(K, dtype=bool)

# #         bnet = self.baselines[t]
# #         tau = self.taus[t]

# #         if self.verbose:
# #             print(f"[guide] t={t} K={K} d={d} tau={tau:.6f}")

# #         while not done.all():
# #             active = np.where(~done)[0]
# #             Ka = active.size
# #             Xprop = super().single_step_iid_copies(t, Ytp1[active], size=batch_size)  # (B,Ka,d)
# #             Xflat = Xprop.reshape(batch_size * Ka, d)

# #             # v_t on proposals
# #             t_arr = (self.T - 1 - t) * np.ones(Xflat.shape[0])
# #             v_vals = self._value_eval_numpy(t_arr, Xflat).reshape(batch_size, Ka)  # (B,Ka)

# #             # log u on prefixes (broadcast over B)
# #             with torch.no_grad():
# #                 ypref = torch.tensor(Ytp1[active], dtype=torch.float32, device=self.device)
# #                 b_vals = bnet(ypref, t+1).detach().cpu().numpy()            # (Ka,)

# #             print("v_vals:", v_vals)
# #             print("logu_vals:", b_vals)

# #             # acceptance exponent: tau + logu(y_{t+1}) - v_t(x_t)
# #             accept_log_exponent = tau + b_vals[None, :] - v_vals            # (B,Ka)
# #             print("accept_log_exponent shape:" , accept_log_exponent)
# #             # clip at 1: exp(min(0, ·))
# #             R = np.exp(np.minimum(0.0, accept_log_exponent))
# #             U = np.random.rand(batch_size, Ka)
# #             accept = (U < R)

# #             if self.verbose:
# #                 print(f"[guide] active={Ka} Xprop.shape={Xprop.shape} v_vals.shape={v_vals.shape} logu_vals.shape={logu_vals.shape}")
# #                 print(f"        accept_log_exponent.shape={accept_log_exponent.shape} R.mean={R.mean():.6f}")

# #             # proposal accounting like your rejection utils
# #             for k in range(Ka):
# #                 idx = np.where(accept[:, k])[0]
# #                 self.num_steps += (idx[0] + 1) if idx.size > 0 else batch_size

# #             got_any = accept.any(axis=0)
# #             if got_any.any():
# #                 first_idx = np.argmax(accept, axis=0)
# #                 sel = active[got_any]
# #                 picks = first_idx[got_any]
# #                 out[sel] = Xprop[picks, got_any, :]
# #                 done[sel] = True

# #         return out

# #     # ---------------- overall backward train+sample ----------------
# #     def train_baselines_and_sample(self, batch_size=1024):
# #         """
# #         1) Initialize y_T ~ N(0, I).
# #         2) For t = T-2 ... 0:
# #            • Train b_{t+1} (log u) + λ_t on { (y_{t+1}^i, x_t^i) } with one x_t per prefix.
# #            • Compute τ_t with ERM slack.
# #            • Sample y_t via rejection using (b_{t+1}, τ_t).
# #         Returns X in chronological order (S, T, d).
# #         """
# #         T = len(self.alpha_t)
# #         X = np.zeros((self.S, T, self.d))
# #         X[:, -1, :] = np.random.normal(size=(self.S, self.d))

# #         for t in tqdm(range(T - 2, -1, -1), desc="Chernoff backward"):
# #             Ytp1 = X[:, t + 1, :]
# #             if self.verbose:
# #                 print(f"[loop] t={t} Ytp1.shape={Ytp1.shape}")

# #             bnet, lam_t, tau_t = self.train_baseline_step(
# #                 t=t, Ytp1=Ytp1, batch_size=batch_size
# #             )
# #             self.baselines[t] = bnet
# #             self.lambdas[t] = lam_t
# #             self.taus[t] = tau_t

# #             X[:, t, :] = self.single_step_guide(t, X[:, t + 1, :])

# #         # flip to chronological (x_0 ... x_T)
# #         return np.flip(X, axis=1)

# #     def calc_samples(self, S=5000, YT=None, batch_size=16, reset_counter=True):
# #         """
# #         Generate samples using ALREADY-TRAINED baselines (log u_{t+1}) and taus.
# #         Includes tqdm progress bars during backward diffusion.
# #         """
# #         # ---- Sanity checks ----
# #         if any(b is None for b in self.baselines):
# #             raise RuntimeError("Baselines not set. Train or load self.baselines before calling sample_only().")
# #         if np.any([np.isnan(t) or np.isinf(t) for t in self.taus]):
# #             raise RuntimeError("Found invalid tau values. Ensure taus are computed before inference.")

# #         # ---- Initialization ----
# #         if YT is not None:
# #             YT = np.asarray(YT)
# #             if YT.ndim != 2 or YT.shape[1] != self.d:
# #                 raise ValueError(f"YT must have shape (S, d); got {YT.shape}.")
# #             S_eff = YT.shape[0]
# #         else:
# #             S_eff = S if S is not None else self.S
# #             YT = np.random.normal(size=(S_eff, self.d))

# #         if reset_counter:
# #             self.num_steps = 0

# #         T = len(self.alpha_t)
# #         X = np.zeros((S_eff, T, self.d), dtype=float)
# #         X[:, -1, :] = YT
# #         print(S)
# #         if self.verbose:
# #             print(f"[sample_only] S={S_eff} T={T} d={self.d}")
# #             print(f"[sample_only] YT.shape={YT.shape}")

# #         # ---- Backward sampling with tqdm ----
# #         for t in tqdm(range(T - 2, -1, -1), desc="Chernoff inference", leave=True):
# #             Ytp1 = X[:, t + 1, :]
# #             if self.verbose:
# #                 print(f"[sample_only] t={t} Ytp1.shape={Ytp1.shape} tau={self.taus[t]:.6f}")
# #             X[:, t, :] = self.single_step_guide(t, Ytp1, batch_size=batch_size)

# #         X_chrono = np.flip(X, axis=1)
# #         if self.verbose:
# #             print(f"[sample_only] done. X_chrono.shape={X_chrono.shape} num_steps={self.num_steps}")

# #         return X_chrono
