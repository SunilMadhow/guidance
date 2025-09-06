
from value_estimator import ValueEstimate
from rejection import *

class Diffusion:
    def __init__(self, dist0, alpha_t, S = 10000):
        self.dist0 = dist0
        self.alpha_t = alpha_t
        self.d = dist0.mu.shape[1]
        self.S = S
        self.alpha_bar = np.cumprod(self.alpha_t)
    
    def calc_samples(self):
        # Initialize a matrix to store samples at each time step
        X = np.zeros((self.S, len(self.alpha_t), self.d))
        # Start from standard Gaussian noise at time T
        X[:, -1, :] = np.random.normal(size=(self.S, self.d))
        # Reverse diffusion from t=T-1 down to t=0
        for t in tqdm(range(len(self.alpha_t) - 2, -1, -1), desc="Reverse Diffusion Progress"):
            X[:, t, :] = self.single_step(t, X[:, t + 1, :])
        return np.flip(X, axis = 1)

    def single_step(self, t, X):
        # print("Diffusion.single_step")
        # print("X.shape", X.shape)
        X_next = (X + (1 - self.alpha_t[t]) * self.dist0.score(X, self.alpha_bar[t])) / np.sqrt(self.alpha_t[t])
        X_next += np.sqrt((1 - self.alpha_t[t]) / self.alpha_t[t]) * np.random.normal(size=(self.S, self.d))
        return X_next
    
    def single_step_iid_copies(self, t, X, size = 1):
        X_next_arr = []
        for _ in range(size):
            X_next = (X + (1 - self.alpha_t[t]) * self.dist0.score(X, self.alpha_bar[t])) / np.sqrt(self.alpha_t[t])
            X_next += np.sqrt((1 - self.alpha_t[t]) / self.alpha_t[t]) * np.random.normal(size=(X.shape[0], self.d))
            X_next_arr.append(X_next)
        return np.array(X_next_arr)    
    
    def single_sample(self, t, x, size = 1):
        x_next = (x + (1 - self.alpha_t[t]) * self.dist0.score(x, self.alpha_bar[t])) / np.sqrt(self.alpha_t[t])
        x_next += np.sqrt((1 - self.alpha_t[t]) / self.alpha_t[t]) * np.random.normal(self.d)
        return x_next


class GuidedDiffusion(Diffusion):
    def __init__(self, dist0, alpha_t, value_function,
                 upper_bound: float = 1.0, temperature: float = 1.0, S: int = 10_000, lower_bound = None):
        super().__init__(dist0, alpha_t, S)
        self.value_function = value_function
        self.upper_bound = upper_bound
        self.temperature = temperature
        self.T = len(self.alpha_t)
        self.B = self.upper_bound / self.temperature
        if lower_bound is not None:
            self.B = 0
        self.num_steps = 0

    def evaluate(self, x, t):        
        return self.value_function(x, self.T - 1 - t)
    
    def single_step(self, t, X):
        N = X.shape[0]
        proposal_fixed = lambda size, src: super(GuidedDiffusion, self).single_step_iid_copies(t, src, size)
        eval_fixed = lambda x: self.evaluate(x, t)
        Z =  vec_rejection_val(X, proposal_fixed, eval_fixed, self.B, batch_size=None)
        self.num_steps += Z[1]
        return Z[0]
    
class LastStepDiffusion(Diffusion):
    def __init__(self, dist0, alpha_t, reward_function, upper_bound : float = 1, temperature : float = 1,  S=10000):
        super().__init__(dist0, alpha_t, S)
        self.reward_function = reward_function
        self.upper_bound = upper_bound
        self.temperature = temperature
        self.B = self.upper_bound / self.temperature
        self.num_steps = 0

    def calc_samples(self):
        # Initialize a matrix to store samples at each time step
        final_samples = []
        with tqdm(total=self.S, desc="Generating Trajectories") as pbar:
            while len(final_samples) < self.S:
                X = np.zeros((self.S, len(self.alpha_t), self.d))
                # Start from standard Gaussian noise at time T
                X[:, -1, :] = np.random.normal(size=(self.S, self.d))
                # Reverse diffusion from t=T-1 down to t=0
                for t in range(len(self.alpha_t) - 2, -1, -1):
                    self.num_steps += self.S
                    X[:, t, :] = self.single_step(t, X[:, t + 1, :])
                # print("X.shape", X.shape)

                # Reject trajectories based on reward function
                for trajectory in X:
                    x_final = trajectory[0]
                    reward = self.reward_function(x_final)
                    acceptance_prob = np.exp(reward -self.B)
                    # print("Acceptance probability:", acceptance_prob)
                    if np.random.uniform(0, 1) < acceptance_prob:
                        final_samples.append(trajectory)
                        pbar.update(1)
                    if len(final_samples) >= self.S:
                        self.num_steps = self.num_steps - 1

        return np.flip(np.array(final_samples[:self.S]), axis = 1)
    
import numpy as np
from tqdm import tqdm
from value_estimator import ValueEstimate          # your wrapper


class MixedGuidance(Diffusion):
    """
    Hybrid sampler:
      • Phase 1 : unconditional roll‑in T … k, global rejection via v_k
      • Phase 2 : guided per‑step rejection  k-1 … 0
    """
    def __init__(self,
                 dist0,
                 alpha_t,
                 value_function,          # callable (x, t) -> v_t(x)x
                 k: int,                 # pivot timestep (1 ≤ k < T)
                 upper_bound: float = 1.,
                 temperature: float = 1.,
                 S: int = 10_000,
                 batch: int | None = None,   # batch size for phase‑1
                 lower_bound=None):
        super().__init__(dist0, alpha_t, S)
        # assert 1 <= k < len(alpha_t)-1, "`k` must be in [1, T-1)"
        self.k               = k
        self.value_function  = value_function
        self.upper_bound     = upper_bound
        self.temperature     = temperature
        self.T              = len(self.alpha_t)
        self.B = self.upper_bound / self.temperature
        if lower_bound is not None:
            self.B = 0.0

        self.batch      = batch or S       # default: one big batch like LastStep
        self.num_steps  = 0                # reverse‑kernel calls (proposal count)

    # ------------------------------------------------------------------
    # helpers

    def evaluate(self, x, t):
        """Value estimate  v_t(x)."""
        # print("evaluate(x) called with x of shape ", x.shape)
        out = self.value_function(x, self.T - 1 - t)
        # print("evaluate(x, t) has shape", out.shape)
        return out

    def single_step_(self, t, X):
        N = X.shape[0]
        proposal_fixed = lambda size, src: super(MixedGuidance, self).single_step_iid_copies(t, src, size)
        eval_fixed = lambda x: self.evaluate(x, t)
        Z =  vec_rejection_val(X, proposal_fixed, eval_fixed, self.B, batch_size=None)
        self.num_steps += Z[1]
        return Z[0]

    # ------------------------------------------------------------------
    # main sampler

    def calc_samples(self):
        """
        Generate `self.S` trajectories exactly according to the mixed‑guidance
        scheme, returning shape (S, T, d) in chronological order (x_0 … x_T).
        """
        T = len(self.alpha_t)
        final_samples = []
        with tqdm(total=self.S, desc="Phase 1 Trajectories") as pbar:
            while len(final_samples) < self.S:
                # Start from standard Gaussian noise at time T
                X = np.zeros((self.S, len(self.alpha_t), self.d))
                X[:, -1, :] = np.random.normal(size=(self.S, self.d))
                # Reverse diffusion from t=T-1 down to t=k
                for t in range(len(self.alpha_t) - 2, self.k, -1):
                    print
                    self.num_steps += self.S
                    X[:, t, :] = self.single_step(t, X[:, t + 1, :])
                # print("X.shape", X.shape)

                # Reject trajectories based on reward function
                for trajectory in X[:,self.k:T,:]:
                    x_final = trajectory[-1]
                    # print("x_final", x_final.shape)
                    reward = self.evaluate(x_final, self.k)
                    # print("reward", reward)
                    acceptance_prob = np.exp(reward -self.B)
                    # print("Acceptance probability:", acceptance_prob)
                    if np.random.uniform(0, 1) < acceptance_prob:
                        final_samples.append(trajectory)
                        pbar.update(1)
                    if len(final_samples) >= self.S: #we didn't really need all of these
                        self.num_steps = self.num_steps - 1
        final_samples = np.array(final_samples[:self.S])
        
        
        X = np.zeros((self.S, len(self.alpha_t), self.d))
        print("final_samples", final_samples.shape)
        print("ayayay", X[:, self.k:T, :].shape)
        X[:, self.k:T, :] = np.array(final_samples)
        

        # # -------- Phase 2 : per‑step rejection  k-1 … 0 ----------
        for t in tqdm(range(self.k, -1, -1), desc="Phase 2 guided rollout"):
            prev   = X[:, t+1, :]
            step = self.single_step_(t, prev)
            X[:, t, :] = step

        # chronological order (x_0 … x_T) to match Diffusion.calc_samples
        return np.flip(X, axis=1)

class TightGuidance(Diffusion):
    def __init__(self, dist0, alpha_t, value_function,
                 upper_bound: float = 1.0, temperature: float = 1.0, S: int = 10_000, lower_bound = None):
        super().__init__(dist0, alpha_t, S)
        self.value_function = value_function
        self.upper_bound = upper_bound
        self.temperature = temperature
        self.T = len(self.alpha_t)
        self.B = self.upper_bound / self.temperature
        if lower_bound is not None:
            self.B = 0
        self.num_steps = 0

    def evaluate(self, x, t):        
        return self.value_function(x, self.T - 1 - t)
    
    def single_step(self, t, X):
        N = X.shape[0]
        proposal_fixed = lambda size, src: super(TightGuidance, self).single_step_iid_copies(t, src, size)
        v1 = lambda x: self.evaluate(x, t)
        # print("X.shape = ", X.shape)
        v2 = self.evaluate(X, t + 1)
        Z =  double_rejection(X, proposal_fixed, v1, v2, batch_size=None)
        self.num_steps += Z[1]
        return Z[0]