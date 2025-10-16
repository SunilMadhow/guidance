import torch
import numpy as np
from tqdm import tqdm
from guided_models import Diffusion

class ChernoffGuidance(Diffusion):
    def __init__(self, dist0, alpha_t, value_fn, delta=1e-3, S=10_000):
        super().__init__(dist0, alpha_t, S)
        self.value_fn = value_fn
        self.delta = delta
        self.S = S
        self.num_steps = 0
        self.T = len(alpha_t)
        self.alpha_bar = np.cumprod(alpha_t)

    def _value_eval(self, t, X):
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
    
    def _baseline_eval(self, t, X):
        X = np.asarray(X)
        n = X.shape[0]
        try:
            out = self.baseline_fn(t, X)
        except TypeError:
            out = self.baseline_fn(np.full((n,), t), X)
        # to numpy, squeeze
        try:
            if isinstance(out, torch.Tensor):
                out = out.detach().cpu().numpy()
        except ImportError:
            pass
        out = np.asarray(out).reshape(n)
        return out
    

    def train_baseline_step(self, t, Xt, Vtm1, n_epochs):
        pass
    
    def single_step_guide(self,t, u_tp1, v_t, tau, Xtp1, batch_size = 10):
        K, d = Xtp1.shape
        X_t = self.single_step_iid_copies(t, Xtp1)
        

        P = np.ones(Xtp1.shape)*t + u_tp1(Xtp1) - v_t(X_t)

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
            u_vals = self._baseline_eval(t_arr, Xflat).reshape(batch_size, Ka)  # (B,Ka)
            # Acceptance: exp( v_t(x) - φ_t(μ) - tail )
            numerator = tau*np.ones((batch_size, Ka)) + u_vals - v_vals
            R       = np.exp(numerator)
            U       = np.random.rand(batch_size, Ka)
            accept  = (U < R)
            for k in range(Ka):
                accepted_indices = np.where(accept[:, k])[0]
                if accepted_indices.size > 0:
                    self.num_steps += accepted_indices[0] + 1
                else:
                    self.num_steps += batch_size
            got_any = accept.any(axis=0)
            if got_any.any():
                first_idx = np.argmax(accept, axis=0)
                sel       = active[got_any]
                picks     = first_idx[got_any]
                out[sel]  = Xprop[picks, got_any, :]
                done[sel] = True

        return out



    def train_baseline(self):
        X = np.zeros((self.S, self.T, self.d))
        X[:, -1, :] = np.random.normal(size=(self.S, self.d))
        for t in tqdm(range(self.T, 0, -1)):
            X_ = self.single_step(t, X[:, t, :])
            V = self.value_fn(X_[:, t-1, :])
            u_func, lamb, tau = self.train_baseline_step(X[:, -1, :], V)
            
            X[:, t-1, :] = sample_
            