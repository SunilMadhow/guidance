import numpy as np
from scipy.stats import norm,multivariate_normal,gaussian_kde
from scipy.integrate import nquad
# from numba import njit
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from kl import KLdivergence
from scipy.stats import wasserstein_distance_nd
from tqdm import tqdm, trange

N = 2
d = 2
dc = 1
P = np.zeros(d)
P[:dc] = 1.  # conditioned on first set of elements
P = np.diagflat(P)
y = np.zeros((1,d))
y[0,:dc] = 1

I = np.eye(d)
# n_value = np.array([100,250,500,750,1000,2000,3000,4000,5000,7500,10000])
# n_value = np.array([25, 50, 200, 250])
# n_value = np.linspace(100,1000,10)
# n_value = np.linspace(25,150,10, dtype=int)
# n_value = np.linspace(25,250,10, dtype=int)
# n_value = np.linspace(10,50,10,dtype=int)**2
n_value = np.linspace(25,1000,20, dtype=int)

np.random.seed(100)

mu_0 = (np.random.rand(N,d)-0.5)*2
# mu_0 = np.array([[-1.5,-1.5],[1.5,1.5]])
# var_0 = np.random.uniform(.7,2,size=N)  # Assumes independent mixture
# Assumes same correlated covariance among mixture components
# A = np.random.rand(d, d)
# Sigma_0 = np.dot(A, A.T) / d
Sigma_0 = np.diagflat([0.1,1])
Std_0 = np.sqrt(np.diag(Sigma_0))
Corr_0 = np.diagflat(1/Std_0).dot(Sigma_0).dot(np.diagflat(1/Std_0))
# aa = np.random.uniform(0.4,0.7)
aa = 0.6
Corr_0[dc:,:dc] = aa
Corr_0[:dc,dc:] = aa
var_0 = np.diagflat(Std_0).dot(Corr_0).dot(np.diagflat(Std_0))
# var_0 = np.dot(A, A.T) / d
# var_0 = np.array([0.5,0.5])
# pi_0 = np.random.uniform(size=N)
pi_0 = np.array([0.4,0.6])
pi_0 = pi_0 / np.sum(pi_0)
print(mu_0)
print(var_0)
print(pi_0)
print(aa)

class Mixture(object):
    def __init__(self, mu_0, var_0, pi_0):
        super(Mixture, self).__init__()
        self.mu = mu_0
        self.pi = pi_0
        if len(var_0.shape) == 1:
            # independent components
            self.dist = [multivariate_normal(mu_0[i],var_0[i] * I) for i in range(N)]
            self.var = np.array([v * I for v in var_0])
        elif len(var_0.shape) == 2:
            # same correlated matrix among components
            self.dist = [multivariate_normal(mu_0[i],var_0) for i in range(N)]
            self.var = np.tile(var_0,(N,1,1))
        elif len(var_0.shape) == 3:
            # general correlated matrix among components
            self.dist = [multivariate_normal(mu_0[i],var_0[i]) for i in range(N)]
            self.var = var_0

    def pdf(self, x):
        pdf_each = np.array([dd.pdf(x) for dd in self.dist])
        return np.average(pdf_each, axis=0, weights=self.pi)

    def score(self, x, alpha_bar):
        ## TODO here
        mu_t = np.sqrt(alpha_bar) * self.mu
        var_t = alpha_bar * self.var + (1-alpha_bar) * I
        dist_t = [multivariate_normal(mu_t[i],var_t[i]) for i in range(N)]
        pdf_each = np.array([dd.pdf(x) for dd in dist_t])
        deriv_exp_each = np.array([ -(x-mu_t[i]).dot(np.linalg.inv(var_t[i])) for i in range(N)])
        return np.average(deriv_exp_each * pdf_each[:,:,np.newaxis], axis=0, weights=self.pi) / np.average(pdf_each, axis=0, weights=self.pi)[:, np.newaxis]

    def score_cond(self, x, P, y, alpha_bar):
        # Assumes independent mixtures
        mu_ty = np.sqrt(alpha_bar) * (self.mu.dot(I-P) + y)
        Sigma_ty = np.array([alpha_bar * (I-P).dot(self.var[i]).dot(I-P) + (1-alpha_bar) * I for i in range(N)])
        dist_t = [multivariate_normal(mu_ty[i],Sigma_ty[i]) for i in range(N)]
        pdf_each = np.array([dd.pdf(x) for dd in dist_t])
        deriv_exp_each = np.array([ -(x-mu_ty[i]).dot(np.linalg.inv(Sigma_ty[i])) for i in range(N)])
        return np.average(deriv_exp_each * pdf_each[:,:,np.newaxis], axis=0, weights=self.pi) / np.average(pdf_each, axis=0, weights=self.pi)[:, np.newaxis]

    # @njit
    def rvs(self, size):
        S = size
        palette = np.zeros((N,S,d))
        y = np.zeros((S,d))
        for i in range(N):
            palette[i,:,:] = self.dist[i].rvs(size=S)
        y = palette[np.random.choice(N, p=self.pi), np.arange(S), :]
        # for i in range(S):
        #     y[i,:] = palette[np.random.choice(N, p=self.pi), i, :]
        return y

def calc_samples_cond_oracle(dist0, P, y, alpha_t, S=10000):
    assert(alpha_t[0] > 0)
    alpha_t_bar = np.cumprod(alpha_t)
    X = np.random.normal(size=(S,d))
    for t in range(len(alpha_t)-1,0,-1):
        # X = (X + (1-alpha_t[t]) * dist0.score(X, alpha_t_bar[t])) / np.sqrt(alpha_t[t]) + np.sqrt((1-alpha_t[t]) / alpha_t[t])* np.random.normal(size=(S,d))
        term1 = (1-alpha_t[t]) * dist0.score_cond(X, P, y, alpha_t_bar[t])
        noise = np.sqrt((1-alpha_t[t]) / alpha_t[t])* np.random.normal(size=(S,d))
        X = (X + term1) / np.sqrt(alpha_t[t]) + noise
    return X

def calc_samples_cond(dist0, P, y, alpha_t, S=10000):
    assert(alpha_t[0] > 0)
    alpha_t_bar = np.cumprod(alpha_t)
    X = np.random.normal(size=(S,d))
    for t in range(len(alpha_t)-1,0,-1):
        # X = (X + (1-alpha_t[t]) * dist0.score(X, alpha_t_bar[t])) / np.sqrt(alpha_t[t]) + np.sqrt((1-alpha_t[t]) / alpha_t[t])* np.random.normal(size=(S,d))
        term1 = (1-alpha_t[t]) * dist0.score(X, alpha_t_bar[t]).dot(I-P)
        term2 = (1-alpha_t[t]) * (np.sqrt(alpha_t_bar[t]) * y - X.dot(P)) / (1 - alpha_t_bar[t])
        noise = np.sqrt((1-alpha_t[t]) / alpha_t[t])* np.random.normal(size=(S,d))
        X = (X + term1 + term2) / np.sqrt(alpha_t[t]) + noise
    return X

def calc_cond_kl_from_mix_samp(dist_1y, X, S=10000):
    Y = dist_1y.rvs(S)
    numer = np.log(dist_1y.pdf(Y))
    denom = np.log(gaussian_kde(X.T, bw_method='scott')(Y.T))
    return np.maximum(np.mean(numer - denom), 0.)

def calc_mse_cond(dist0, P, alpha_t, S=10000):
    assert(alpha_t[0] > 0)
    X0 = dist0.rvs(S)
    y = X0.dot(P)
    X1 = np.sqrt(alpha_t[0]) * X0 + np.sqrt(1-alpha_t[0]) * np.random.normal(size=(S,d))
    X1hat = calc_samples_cond(dist0, P, y, alpha_t, S=S)
    return np.average((X1-X1hat)**2)

kl1 = np.zeros(n_value.shape)
kl2 = np.zeros(n_value.shape)
mse = np.zeros(n_value.shape)
dist0 = Mixture(mu_0, var_0, pi_0)
# print(dist0.pdf(dist0.rvs(10000)).shape)
# print(gaussian_kde(dist0.rvs(1000).T)(dist0.rvs(1000).T).shape)
# exit(0)

# gm = GaussianMixture(n_components=N,
#                      covariance_type='spherical',
#                      weights_init=dist0.pi,
#                      means_init=dist0.mu,
#                      precisions_init=1/dist0.var
#                      ).fit(dist0.rvs(10000))
# print(gm.weights_)
# print(gm.means_)
# print(gm.covariances_)
# exit(0)

# x,y = np.mgrid[-5:5:.1, -5:5:.1]
# plt.contourf(x, y, dist0.pdf(np.dstack((x,y))))
# X = dist0.rvs(1000)
# plt.scatter(X[:,0],X[:,1],s=4,c='k')


for i, n in tqdm(enumerate(n_value), total=len(n_value)):
    # n = n_value[i]
    # print(n)

    # Sampling
    t_values = np.arange(1, n + 1)
    c = 4
    delta = 0.02
    inner = delta * (1 + c * np.log(n) / n)**(t_values)
    alpha_t = 1 - c * np.log(n) / n * np.minimum(inner,1)
    alpha_t[0] = 1-delta

    # beta_start = 1e-4
    # beta_end = 3. * np.log(n) / n
    # beta_t = np.linspace(beta_start, beta_end, n)
    # alpha_t = 1 - beta_t

    while kl1[i] == 0.:
        # alpha_t = np.repeat(1 - c * np.log(n) / n, n)
        # print("Start sample generation...")
        X1 = calc_samples_cond(dist0, P, y, alpha_t, 100000)
        # X2 = calc_samples_cond_oracle(dist0, P, y, alpha_t, 40000)
        # gm = GaussianMixture(n_components=N,
        #                      covariance_type='spherical',
        #                      weights_init=dist0.pi,
        #                      means_init=dist0.mu,
        #                      precisions_init=1/dist0.var
        #                      ).fit(X)
        # print(gm.weights_)
        # print(gm.means_)
        # print(gm.covariances_)
        # print("")

        # Evaluating KL
        print("Start KL estimation...")
        mu_1y = np.sqrt(alpha_t[0])*(mu_0.dot(I-P) + y)
        Sigma_1y = np.array([alpha_t[0] * (I-P).dot(dist0.var[j] * I).dot(I-P) + (1-alpha_t[0]) * I for j in range(N)])
        dist_1y = Mixture(mu_1y, Sigma_1y[0], dist0.pi)
        Y = dist_1y.rvs(100000)
        # # kl1[i] = wasserstein_distance_nd(X1, Y)
        kl1[i] = calc_cond_kl_from_mix_samp(dist_1y, X1, 100000)
    # print(KLdivergence(Y, X1))
    print(kl1[i])
    # # kl2[i] = calc_cond_kl_from_mix_samp(dist_1y, X2, 50000)

    # Evaluating MSE
    # print("Start MSE estimation...")
    # mse[i] = calc_mse_cond(dist0, P, alpha_t, S=100000)


    # xgrid,ygrid = np.mgrid[-5:5:.1, -5:5:.1]
    # plt.contourf(xgrid, ygrid, dist_1y.pdf(np.dstack((xgrid,ygrid))))
    # plt.scatter(X[:,0],X[:,1],s=4,c='r')
    # plt.show()
    # exit(0)
print(kl1)
# print(mse)
# print(kl2)
plt.plot(n_value, kl1, 'o--')
# plt.plot(n_value, mse, 'o--')
# plt.plot(n_value, -np.log(kl2), label="oracle")
plt.xlabel(r"$T$")
plt.ylabel(r"$KL(Q_{1|y}||P_{1|y})$")
# plt.ylabel(r"$MSE(Q_{1|y}, P_{1|y})$")
# plt.title(r"$Q_0$ Mixture of indep Gaussian")
# plt.legend(loc="upper left")
plt.show()




###
