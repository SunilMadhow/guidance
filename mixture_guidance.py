import numpy as np
from tqdm import tqdm
from guided_models import Diffusion

class MixtureSpecificGuidance(Diffusion):
    """
    Mixture-aware conditional rejection for DDPM reverse steps.

    Acceptance probability (using a soft value function v):
        p_acc = exp( v_t(x_t) - φ_t( μ_{t|t+1}(x_{t+1}) )
                     - L_r * max_k ||A_t^k||_2 * sqrt(2 * β_{t+1} * log(1/δ)) ).
    """

    def __init__(self, dist0, alpha_t, reward_fn, L_r, value_fn,
                 delta=1e-3, S=10_000, use_trace=True):
        """
        dist0: Mixture object with fields:
              • mu:  (N,d) component means
              • var: (N,d,d) or (d,d) covariance(s)
              • pi:  (N,) mixture weights
              and a score(X, alpha_bar_t) method for the base proposal.
        alpha_t: array of α_t in (0,1].
        reward_fn: callable r(x0) returning shape (n,)
        L_r: Lipschitz constant for r (‖∇r‖ ≤ L_r)
        value_fn: callable v(t, x) -> shape (n,) for x shape (n,d)
        delta: tail probability for the high-probability bound
        S: batch size / #trajectories
        use_trace: if True uses trace(Σ), else spectral ‖Σ‖₂ in the moment bound
        """
        super().__init__(dist0, alpha_t, S)
        self.reward_fn   = reward_fn
        self.L_r         = float(L_r)
        self.delta       = float(delta)
        self.use_trace   = bool(use_trace)
        self.value_fn    = value_fn
        self.num_steps = 0

        # Normalize covariance format to (N,d,d)
        self.Nc = dist0.mu.shape[0]
        self.d  = dist0.mu.shape[1]
        if len(dist0.var.shape) == 2:
            self.Sigma = np.tile(dist0.var[None, :, :], (self.Nc, 1, 1))
        elif len(dist0.var.shape) == 3:
            self.Sigma = dist0.var
        else:
            raise ValueError("dist0.var must be (d,d) or (N,d,d).")

        self.mu_comp = dist0.mu
        self.pi      = dist0.pi / np.sum(dist0.pi)

        # Precompute ᾱ_t
        self.alpha_bar = np.cumprod(self.alpha_t)
        self.T = len(self.alpha_t)

    # ---------- utilities ----------
    def _chol_inv_and_logdet(self, S):
        """Return (S^{-1}, log|S|) using Cholesky; S is (d,d)."""
        L = np.linalg.cholesky(S)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        Linv  = np.linalg.inv(L)
        Sinv  = Linv.T @ Linv
        return Sinv, logdet

    def _logpdf_gauss(self, x, m, S_inv, logdetS):
        """Return log N(x; m, S) for x (K,d), m (d,), S_inv (d,d)."""
        K = x.shape[0]
        diff = x - m[None, :]
        qf = np.einsum('ki,ij,kj->k', diff, S_inv, diff)
        return -0.5 * (self.d*np.log(2*np.pi) + logdetS + qf)

    def _mixture_time_covs(self, t):
        """
        For time t: m_t^k = √ᾱ_t μ_k,  Σ_t^k = ᾱ_t Σ_k + (1-ᾱ_t) I
        Also A_t^k = √ᾱ_t Σ_k (Σ_t^k)^{-1}  (used in Lipschitz of φ_t).
        """
        a_bar = self.alpha_bar[t]
        I = np.eye(self.d)
        m_t = np.sqrt(a_bar) * self.mu_comp                       # (N,d)
        Sigma_t = a_bar * self.Sigma + (1.0 - a_bar) * I[None]    # (N,d,d)

        A_list, Sinv_list, logdet_list = [], [], []
        for k in range(self.Nc):
            Sinv_k, logdet_k = self._chol_inv_and_logdet(Sigma_t[k])
            A_k = np.sqrt(a_bar) * self.Sigma[k].dot(Sinv_k)
            A_list.append(A_k)
            Sinv_list.append(Sinv_k)
            logdet_list.append(logdet_k)
        A = np.stack(A_list)                 # (N,d,d)
        Sinv = np.stack(Sinv_list)           # (N,d,d)
        logdet = np.array(logdet_list)       # (N,)
        return m_t, Sigma_t, A, Sinv, logdet

    def _responsibilities(self, t, x_t):
        """γ_t^k(x_t) for all samples (K,d)."""
        x_t = np.asarray(x_t)
        K = x_t.shape[0]
        m_t, Sigma_t, _, Sinv, logdet = self._mixture_time_covs(t)
        logw = []
        for k in range(self.Nc):
            lp = np.log(self.pi[k]) + self._logpdf_gauss(x_t, m_t[k], Sinv[k], logdet[k])
            logw.append(lp)
        logw = np.stack(logw, axis=1)  # (K,N)
        m = np.max(logw, axis=1, keepdims=True)
        w = np.exp(logw - m)
        gamma = w / np.sum(w, axis=1, keepdims=True)
        return gamma  # (K,N)

    def _posterior_x0_given_xt(self, t, x_t):
        """
        Returns per-sample arrays:
          μ0t: (K,N,d), Σ0t: (N,d,d) [Σ0|t is independent of x_t], γ: (K,N)
        with formulas:
          μ0|t^k = μ_k + √ᾱ_t Σ_k (Σ_t^k)^{-1} (x_t - √ᾱ_t μ_k)
          Σ0|t^k = Σ_k − ᾱ_t Σ_k (Σ_t^k)^{-1} Σ_k
        """
        x_t = np.asarray(x_t)
        K = x_t.shape[0]
        a_bar = self.alpha_bar[t]
        m_t, Sigma_t, A, Sinv, _ = self._mixture_time_covs(t)

        mu0_list, Sigma0_list = [], []
        for k in range(self.Nc):
            Ak = A[k]  # √ᾱ Σ_k (Σ_t^k)^{-1}
            mu0_k = (x_t @ Ak.T) + (self.mu_comp[k] - Ak @ m_t[k])  # (K,d)
            mu0_list.append(mu0_k)
            Sigma0_k = self.Sigma[k] - a_bar * self.Sigma[k].dot(Sinv[k]).dot(self.Sigma[k])
            Sigma0_list.append(Sigma0_k)

        mu0t = np.stack(mu0_list, axis=1)         # (K,N,d)
        Sigma0t = np.stack(Sigma0_list, axis=0)   # (N,d,d)
        gamma = self._responsibilities(t, x_t)    # (K,N)
        return mu0t, Sigma0t, gamma

    def _phi_t(self, t, x_t):
        """
        ϕ_t(x_t) = log Σ_k γ_t^k(x_t) * exp{ r(μ0|t^k) + (L_r^2/2) * M(Σ0|t^k) }
        with M(Σ) = trace(Σ)  (or spectral norm if use_trace=False).
        Returns (K,)
        """
        K = x_t.shape[0]
        mu0t, Sigma0t, gamma = self._posterior_x0_given_xt(t, x_t)   # (K,N,d), (N,d,d), (K,N)
        r_vals = self.reward_fn(mu0t.reshape(K*self.Nc, self.d)).reshape(K, self.Nc)
        if self.use_trace:
            M = np.array([np.trace(S) for S in Sigma0t])            # (N,)
        else:
            M = np.array([np.linalg.norm(S, 2) for S in Sigma0t])   # (N,)
        bump = 0.5 * (self.L_r**2) * M[None, :]                     # (K,N)
        log_terms = np.log(gamma + 1e-40) + r_vals + bump           # (K,N)
        m = np.max(log_terms, axis=1, keepdims=True)
        phi = m.squeeze(1) + np.log(np.sum(np.exp(log_terms - m), axis=1) + 1e-40)  # (K,)
        return phi

    def _mu_t_given_tplus1(self, t, x_tp1):
        """
        μ_{t|t+1}(x_{t+1}) = ( x_{t+1} + (1-α_t) s_{t+1}(x_{t+1}) ) / √α_t
        where s_{t+1} is score at ᾱ_{t+1}.
        """
        s_tp1 = self.dist0.score(x_tp1, self.alpha_bar[t+1])     # (K,d)
        return (x_tp1 + (1.0 - self.alpha_t[t]) * s_tp1) / np.sqrt(self.alpha_t[t])

    def _value_eval(self, t, X):
        """
        Evaluate v_t(X) robustly:
          - try value_fn(t, X)
          - if needed, try value_fn(np.full(n, t), X)
          - accept torch or numpy outputs; return (n,) numpy
        """
        X = np.asarray(X)
        n = X.shape[0]
        try:
            out = self.value_fn(t, X)
        except TypeError:
            out = self.value_fn(np.full((n,), t), X)
        # to numpy, squeeze
        try:
            import torch
            if isinstance(out, torch.Tensor):
                out = out.detach().cpu().numpy()
        except ImportError:
            pass
        out = np.asarray(out).reshape(n)
        return out

    def evaluate(self, x, t):        
        return self.value_fn(x, self.T - 1 - t)
    # ---------- main guided step ----------
    def single_step(self, t, Xtp1, init_batch=None):
        """
        Rejection sampling with acceptance using v_t(·) vs φ_t(μ_{t|t+1}) and the Lipschitz tail.
        """
        Xtp1 = np.asarray(Xtp1)
        K, d = Xtp1.shape
        assert d == self.d

        # Precompute per-chain constants
        mu      = self._mu_t_given_tplus1(t, Xtp1)         # (K,d)
        phi_mu  = self._phi_t(t, mu)                       # (K,)
        _, _, A, _, _ = self._mixture_time_covs(t)         # A_t^k
        Lphi    = self.L_r * max(np.linalg.norm(Ak, 2) for Ak in A)
        beta    = (1.0 - self.alpha_t[t]) / self.alpha_t[t]
        tail    = Lphi * np.sqrt(2.0 * beta * np.log(1.0 / self.delta))

        # modest default; you can adaptively tune like before if desired
        batch_size = init_batch or 5

        out  = np.empty((K, d))
        done = np.zeros(K, dtype=bool)

        while not done.all():
            active = np.where(~done)[0]
            Ka = active.size

            # Draw proposals: shape (batch, Ka, d)
            # print("Xtp1 shape = ", Xtp1.shape)

            Xprop = super().single_step_iid_copies(t, Xtp1[active], size=batch_size)
            Xflat = Xprop.reshape(batch_size*Ka, d)

            # v_t on proposals
            t_arr = [self.T - 1 - t] * (batch_size * Ka)
            v_vals  = self._value_eval(t_arr, Xflat).reshape(batch_size, Ka)      # (B,Ka)

            # Acceptance: exp( v_t(x) - φ_t(μ) - tail )
            numer   = v_vals - phi_mu[active][None, :] - tail
            R       = np.exp(numer)
            R       = np.clip(R, 0.0, 1.0)
            U       = np.random.rand(batch_size, Ka)
            accept  = (U < R)
            for k in range(Ka):
                accepted_indices = np.where(accept[:, k])[0]
                if accepted_indices.size > 0:
                    # First accepted proposal index (inclusive)
                    self.num_steps += accepted_indices[0] + 1
                else:
                    # No accepted proposal, all proposals tested
                    self.num_steps += batch_size
            got_any = accept.any(axis=0)
            if got_any.any():
                first_idx = np.argmax(accept, axis=0)
                sel       = active[got_any]
                picks     = first_idx[got_any]
                out[sel]  = Xprop[picks, got_any, :]
                done[sel] = True

        return out

    def calc_samples(self):
        T = len(self.alpha_t)
        X = np.zeros((self.S, T, self.d))
        # start from standard normal at t = T-1
        X[:, -1, :] = np.random.normal(size=(self.S, self.d))
        for t in tqdm(range(T - 2, -1, -1), desc="Mixture reverse"):
            X[:, t, :] = self.single_step(t, X[:, t + 1, :])
        # return in chronological order (x_0 ... x_T-1)
        return np.flip(X, axis=1)
