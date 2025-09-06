import numpy as np
from scipy.stats import norm, multivariate_normal, gaussian_kde
from scipy.integrate import nquad
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from guided_models import GuidedDiffusion, LastStepDiffusion

# Set dimensionality and number of mixture components
d = 2          # data dimension
N = 2          # number of Gaussian components
Nkern = 10 # number of samples for KDE/MC integration

# Identity matrix in d dimensions
I = np.eye(d)

# A range of "time horizons" T to test
# n_value = np.array([30, 50, 75, 100, 150, 200, 250, 300, 400, 500])
# n_value = np.array([100])

# Fix random seed for reproducibility
np.random.seed(10)

# Initialize random mixture means in [-1,1]^d
mu_0 = np.array([[5, 0], [-5, 0]])

# Build a random positive-definite covariance by AA^T/d
# a = np.random.rand(d, d) * 2
# var_0 = np.dot(a, a.T) / d   # shared covariance for each component
var_0 = np.eye(d)

# Mixture weights (normalized)
pi_0 = np.array([0.5, 0.5])
pi_0 /= np.sum(pi_0)



from rejection import vec_rejection_val

class Mixture(object):
    """
    Represents an N-component Gaussian mixture model.

    Attributes:
        mu (ndarray): shape (N, d) component means
        var (ndarray): covariances, shape depends on input (see below)
        pi (ndarray): mixture weights, shape (N,)
        dist (list): list of scipy.stats multivariate_normal objects
    """
    def __init__(self, mu_0, var_0, pi_0):
        super(Mixture, self).__init__()
        self.mu = mu_0
        self.pi = pi_0
        # Handle three cases for var_0 input shape:
        if len(var_0.shape) == 1:
            # Diagonal covariances provided as vector
            self.dist = [multivariate_normal(mu_0[i], var_0[i] * I)
                         for i in range(N)]
            self.var = np.array([v * I for v in var_0])
        elif len(var_0.shape) == 2:
            # Single shared covariance
            self.dist = [multivariate_normal(mu_0[i], var_0)
                         for i in range(N)]
            # Tile to shape (N, d, d)
            self.var = np.tile(var_0, (N, 1, 1))
        elif len(var_0.shape) == 3:
            # Each component has its own covariance matrix
            self.dist = [multivariate_normal(mu_0[i], var_0[i])
                         for i in range(N)]
            self.var = var_0

    def pdf(self, x):
        """
        Compute mixture density at points x.

        Args:
            x: array of shape (..., d)
        Returns:
            pdf: array of same leading shape, density values
        """
        # Evaluate each component pdf
        pdf_each = np.array([dd.pdf(x) for dd in self.dist])
        # Weighted average by mixture weights
        return np.average(pdf_each, axis=0, weights=self.pi)

    def score(self, x, alpha_bar):
    # --- ensure x is a 2‐D array of shape (n_pts, d) ----
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]    # now (1, d)

        n_pts, d = x.shape
        mu_t  = np.sqrt(alpha_bar) * self.mu      # (N, d)
        var_t = alpha_bar * self.var + (1 - alpha_bar) * I  # (N, d, d)
        inv_var_t = np.linalg.inv(var_t)          # (N, d, d)
        N = len(self.pi)

        # --- build frozen distributions (optional) ----
        dist_t = [multivariate_normal(mu_t[i], var_t[i]) for i in range(N)]

        # --- compute pdf_each by looping over samples ----
        # pdf_each[i, j] = p(component i at sample j)
        pdf_each = np.empty((N, n_pts))
        for i, dist in enumerate(dist_t):
            for j in range(n_pts):
                pdf_each[i, j] = dist.pdf(x[j])

        # --- compute the “expected gradient” per component/sample ----
        # deriv_exp[i, j, :] = ∇ log p_i(x_j)
        deriv_exp = np.empty((N, n_pts, d))
        for i in range(N):
            for j in range(n_pts):
                diff = x[j] - mu_t[i]
                deriv_exp[i, j] = - diff.dot(inv_var_t[i])

        # --- weighted average over components ----
        # numerator: E_pi[ deriv * p ]    shape => (n_pts, d)
        num = np.tensordot(self.pi, deriv_exp * pdf_each[..., None], axes=(0, 0))
        # denominator: E_pi[ p ]          shape => (n_pts,)
        den = np.dot(self.pi, pdf_each)           # (n_pts,)

        # final score: (num/den) per sample
        score = num / den[:, None]                # (n_pts, d)

        # if user originally passed in a 1‑D x, you might want to return a (d,) array:
        return score[0] if score.shape[0] == 1 else score


    def hess(self, x, alpha_bar):
        """
        Compute the Hessian (second derivative) of log-density at x.

        Args:
            x: array (num_points, d)
            alpha_bar: noise schedule cumulative product
        Returns:
            hess: shape (num_points, d, d)
        """
        npts = x.shape[0]
        mu_t = np.sqrt(alpha_bar) * self.mu
        var_t = alpha_bar * self.var + (1 - alpha_bar) * I
        # Precompute inverses
        var_inv = np.array([np.linalg.inv(v) for v in var_t])

        # Component pdfs
        pdf_each = np.array([
            multivariate_normal(mu_t[i], var_t[i]).pdf(x)
            for i in range(N)
        ])

        # First derivative term (score)
        deriv_exp = np.array([
            -(x - mu_t[i]).dot(var_inv[i])
            for i in range(N)
        ])  # (N, npts, d)
        score_t = np.average(deriv_exp * pdf_each[:, :, None], axis=0, weights=self.pi) / np.average(pdf_each, axis=0, weights=self.pi)[:, None]

        # Second derivative term (expected outer product minus inv covariance)
        sec_deriv = np.empty((N, npts, d, d))
        for i in range(N):
            delta = (x - mu_t[i]).dot(var_inv[i])
            # Outer(delta, delta) - inv(cov)
            sec_deriv[i] = delta[:, None, :] * delta[:, :, None] - var_inv[i]

        # Weighted average and subtract score outer product
        term = np.average(sec_deriv * pdf_each[:, :, None, None], axis=0, weights=self.pi)
        term /= np.average(pdf_each, axis=0, weights=self.pi)[:, None, None]
        return term - (score_t[:, :, None] * score_t[:, None, :])

    def rvs(self, size):
        """
        Sample from the mixture model.

        Args:
            size: number of samples to draw
        Returns:
            y: array (size, d)
        """
        # Draw from each component independently
        palette = np.zeros((N, size, d))
        for i in range(N):
            palette[i] = self.dist[i].rvs(size=size)
        # Choose components according to mixture weights
        idx = np.random.choice(N, size=size, p=self.pi)
        return palette[idx, np.arange(size)]
from tqdm import tqdm

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
        X_next = (X + (1 - self.alpha_t[t]) * self.dist0.score(X, self.alpha_bar[t])) / np.sqrt(self.alpha_t[t])
        X_next += np.sqrt((1 - self.alpha_t[t]) / self.alpha_t[t]) * np.random.normal(size=(self.S, d))
        return X_next
    
    def single_step_iid_copies(self, t, X, size = 1):
        X_next_arr = []
        for _ in range(size):
            X_next = (X + (1 - self.alpha_t[t]) * self.dist0.score(X, self.alpha_bar[t])) / np.sqrt(self.alpha_t[t])
            X_next += np.sqrt((1 - self.alpha_t[t]) / self.alpha_t[t]) * np.random.normal(size=(X.shape[0], d))
            X_next_arr.append(X_next)
        return np.array(X_next_arr)    
    
    def single_sample(self, t, x, size = 1):
        x_next = (x + (1 - self.alpha_t[t]) * self.dist0.score(x, self.alpha_bar[t])) / np.sqrt(self.alpha_t[t])
        x_next += np.sqrt((1 - self.alpha_t[t]) / self.alpha_t[t]) * np.random.normal(d)
        return x_next

    


    
def reward_func(x, means, var, goodmeans, r = 1): #assumes all mixtures have same covariance matrix
    for mean_idx in goodmeans:
        mean = means[mean_idx]
        Z_score = np.sqrt((x - mean).T @ np.linalg.inv(var) @ (x - mean))
        if Z_score <= 4:
            return r
    return -r

if __name__ == "__main__":
    X = None    
    n = n_value[0]
    t_vals = np.arange(1, n+1)
    c, delta = 4, 0.02
    inner = delta * (1 + c * np.log(n) / n)**t_vals
    alpha_t = 1 - (c * np.log(n) / n) * np.minimum(inner, 1)
    alpha_t[0] = 1 - delta
    # Initialize diffusion process
    dist0 = Mixture(mu_0, var_0, pi_0)
    diffusion = Diffusion(dist0, alpha_t, Nkern)
    # Generate samples
    X = diffusion.calc_samples()
    expr = np.exp([reward_func(x, mu_0, var_0, [1]) for x in X[:, -1, :]])
    print("Successfully generated samples.")


    learned_guided_diffusion = LearnedGuidedDiffusion(
        dist0=dist0,
        alpha_t=alpha_t,
        data_X=X,
        data_expr=expr,
        S = 100,
        n_epochs = 1,
        upper_bound=1.0,
        temperature=0.1
    )

    print("calculating guided samples")

    X_guided = learned_guided_diffusion.guided_diffusion.calc_samples()

    # Scatter plot of X projected into two dimensions

    # Visualize the density of dist0 and superimpose the scatter plot of samples
    x = np.linspace(-7, 7, 100)
    y = np.linspace(-7, 7, 100)
    X_grid, Y_grid = np.meshgrid(x, y)
    grid_points = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=-1)
    Z = dist0.pdf(grid_points).reshape(X_grid.shape)

    # Plot histogram of expr
    # plt.figure()
    # plt.hist(expr, bins=50, color="blue", alpha=0.7, edgecolor="black")
    # plt.title("Histogram of expr")
    # plt.xlabel("expr values")
    # plt.ylabel("Frequency")
    # plt.grid(True)

    plt.contourf(X_grid, Y_grid, Z, levels=50, cmap="viridis", alpha=0.8)
    plt.colorbar(label="Density")
    plt.title("Density of dist0 with Samples")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")


    if X is not None:
        plt.scatter(X[:,-1, 0], X[:,-1, 1], alpha=0.3, s=10, color="white", label="Samples")
        plt.legend()

    if X_guided is not None:
        plt.scatter(X_guided[:,-1, 0], X_guided[:,-1, 1], alpha=0.3, s=10, color="yellow", label="Samples")
        plt.legend()

    # Plot histogram of value function for all t
    plt.figure(figsize=(10, 6))
    all_values = []
    for t in range(X_guided.shape[1]):
        values = learned_guided_diffusion.value_func(X_guided[:, t, :], t)
        all_values.extend(values)

    plt.hist(all_values, bins=50, color="green", alpha=0.7, edgecolor="black")
    plt.title("Histogram of Learned Value Function Across All Timesteps")
    plt.xlabel("Value Function Output")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.show()