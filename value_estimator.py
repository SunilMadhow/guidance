import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class TrajectoryDataset(Dataset):
    def __init__(self, X, expr):
        """
        X: tensor (n, T, d)
        expr: tensor (n,)
        """
        self.X = X
        self.expr = expr
        self.n, self.T, self.d = X.shape

    def __len__(self):
        return self.n * self.T

    def __getitem__(self, idx):
        i = idx // self.T
        t = idx % self.T
        x_t = self.X[i, t]                               # (d,)
        time_feat = torch.tensor([t / (self.T - 1)], 
                                 dtype=x_t.dtype)       # scalar in [0,1]
        inp = torch.cat([x_t, time_feat])               # (d+1,)
        return inp, self.expr[i]

class ValueEstimate(nn.Module):
    def __init__(self,
                 X: torch.Tensor,
                 expr: torch.Tensor,
                 hidden_sizes=(64,64),
                 batch_size=128,
                 lr=1e-3,
                 device=None):
        """
        Builds:
          - the MLP v: R^{d+1} -> R
          - a TrajectoryDataset + DataLoader
          - Adam optimizer & MSE loss
        """
        print("Building ValueEstimate...")
        super().__init__()
        if isinstance(X, np.ndarray):
            X = X.copy()
            X = torch.from_numpy(X).float()
        if isinstance(expr, np.ndarray):
            expr = expr.copy()
            expr = torch.from_numpy(expr).float()
        # infer dims
        self.n, self.T, self.d = X.shape
        self.device    = device or (torch.device("cuda") 
                                    if torch.cuda.is_available() 
                                    else torch.device("cpu"))
        # model
        layers = []
        in_dim = self.d + 1
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)
        
        # data & training
        self.dataset    = TrajectoryDataset(X, expr)
        self.loader     = DataLoader(self.dataset, 
                                     batch_size=batch_size, 
                                     shuffle=True)
        self.optimizer  = optim.SGD(self.net.parameters(), lr=lr)
        self.criterion  = nn.MSELoss()
        
        # move to device
        self.to(self.device)

    def forward(self, inp):
        # input shape (..., d+1)
        return self.net(inp).squeeze(-1)

    def train(self, epochs: int = 20, verbose: bool = True):
        """
        Runs `epochs` passes over the data, printing avg MSE each epoch.
        """
        for epoch in range(1, epochs+1):
            self.net.train()  
            total_loss = 0.0
            for inp, target in self.loader:
                inp, target = inp.to(self.device), target.to(self.device)
                pred = self.forward(inp)            # (...,)
                loss = self.criterion(pred, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * inp.size(0)
            avg_mse = total_loss / len(self.dataset)
            if verbose:
                print(f"[Epoch {epoch:02d}] avg MSE: {avg_mse:.4f}")
        return avg_mse
    
    def predict(self, x, t, requires_grad = False):
        """
        Predict values for vectorized input.
        Args:
            x: array-like or tensor of shape (n, d) where n is the number of samples
            t: array-like or tensor of shape (n,) corresponding to time values
        Returns:
            predictions: tensor of shape (n,) containing predicted values
        """
        # Convert x and t to tensors if they are not already
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        elif not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()
        elif not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)

        # Ensure x and t are on the correct device
        x = x.to(self.device)
        t = t.to(self.device)

        time_feat = t.unsqueeze(-1) / (self.T - 1)
        

        # print("x shape:", x.shape)
        # print("time_feat shape:", time_feat.shape)
        inp = torch.cat([x, time_feat], dim=-1)

        self.net.eval()
        if requires_grad:
            predictions = self.forward(inp)
            return predictions
        else:
            with torch.no_grad():
                predictions = self.forward(inp)
            return predictions.numpy()
        
    def predict_under_log(self, x, t, requires_grad = False):
            """
            Predict values for vectorized input.
            Args:
                x: array-like or tensor of shape (n, d) where n is the number of samples
                t: array-like or tensor of shape (n,) corresponding to time values
            Returns:
                predictions: tensor of shape (n,) containing predicted values
            """
            # Convert x and t to tensors if they are not already
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            elif not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)

            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t).float()
            elif not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.float32)

            # Ensure x and t are on the correct device
            x = x.to(self.device)
            t = t.to(self.device)

            time_feat = t.unsqueeze(-1) / (self.T - 1)
            inp = torch.cat([x, time_feat], dim=-1)

            self.net.eval()
            if requires_grad:
                predictions = self.forward(inp)
                preds_pos = F.softplus(predictions) + 1e-6   # POSITIVE & SMOOTH
                return torch.log(preds_pos)
            else:
                with torch.no_grad():
                    predictions = self.forward(inp)
                    preds_pos = F.softplus(predictions) + 1e-6
                return np.log(preds_pos.cpu().numpy())
    


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

class ValueEnsemble(nn.Module):
    """
    Ensemble of T independent regressors.
    Each net t learns to predict expR[:, 0] from X[:, t, :].
    """
    def __init__(self,
                 X,                    # (n, T, d), array or Tensor
                 expr,                 # (n, 1), array or Tensor
                 hidden_sizes=(64, 64),
                 batch_size=128,
                 lr=1e-3,
                 device=None):
        super().__init__()

        # 1) DEVICE
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
                                else torch.device("cpu")
        )
        if isinstance(X, np.ndarray):
            X = X.copy()
            X = torch.from_numpy(X).float()
        if isinstance(expr, np.ndarray):
            expr = expr.copy()
            expr = torch.from_numpy(expr).float()

        n, self.T, d = X.shape
        y = expr.view(-1)   # (n,)

        self.criterion = nn.MSELoss()
        self.nets       = nn.ModuleList()
        self.loaders    = []
        self.optimizers = []

        for t in range(self.T):
            # 3) SLICE + FORCE CONTIGUOUS
            X_t = X[:, t, :].contiguous()  # now guaranteed  (n, d)

            # 4) BUILD DATASET & DATALOADER
            ds     = TensorDataset(X_t, y)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
            self.loaders.append(loader)

            # 5) BUILD MLP FOR TIME‑STEP t
            layers = []
            in_dim = d
            for h in hidden_sizes:
                layers += [nn.Linear(in_dim, h),
                           nn.ReLU(inplace=True)]
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            net = nn.Sequential(*layers).to(self.device)

            self.nets.append(net)
            self.optimizers.append(optim.SGD(net.parameters(), lr=lr))

    def train(self, epochs: int = 20, verbose: bool = False):
        for t, (net, loader, opt) in enumerate(zip(self.nets,
                                                  self.loaders,
                                                  self.optimizers)):
            net.to(self.device)
            for epoch in range(1, epochs + 1):
                net.train()
                total_loss = 0.0
                for x_batch, y_batch in loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    pred = net(x_batch).squeeze(-1)
                    loss = self.criterion(pred, y_batch)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total_loss += loss.item() * x_batch.size(0)

                avg_mse = total_loss / len(loader.dataset)
                if verbose:
                    print(f"[Net {t} | Epoch {epoch:02d}] avg MSE = {avg_mse:.4f}")

    def predict(self, x, t: int):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        # print(f"x shape: {x.shape}")
        net = self.nets[t]
        net.eval()
        with torch.no_grad():
            predictions = net.forward(x)        # Shape (n,)
        # print("output shape:", predictions.shape)
        return predictions.numpy()
    
    def save(self, filepath):
        state = {
            'nets': [net.state_dict() for net in self.nets],
            'optimizers': [opt.state_dict() for opt in self.optimizers],
        }
        torch.save(state, filepath)

    @classmethod
    def load(cls, filepath, X, expr, hidden_sizes=(64, 64), batch_size=128, lr=1e-3, device=None):
        instance = cls(X, expr, hidden_sizes, batch_size, lr, device)
        state = torch.load(filepath, map_location=device)
        for net, net_state in zip(instance.nets, state['nets']):
            net.load_state_dict(net_state)
        for opt, opt_state in zip(instance.optimizers, state['optimizers']):
            opt.load_state_dict(opt_state)
        return instance
    
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm
from sklearn.svm import SVC
class KernelEnsemble:
    """
    Ensemble of T independent kernel machines.
    Each kernel machine t learns to predict expr[:, 0] from X[:, t, :].
    """
    def __init__(self,
                    X,                    # (n, T, d), array or Tensor
                    expr,                 # (n, 1), array or Tensor
                    kernel='rbf',
                    alpha=1.0,
                    gamma=None):
        if isinstance(X, torch.Tensor):
            self.X = X.cpu().numpy()
        else:
            self.X = X
        if isinstance(expr, torch.Tensor):
            self.expr = expr.cpu().numpy()
        else:
            self.expr = expr
        

        self.T = X.shape[1]
        self.models = []
        self.scalers = []

        for t in range(self.T):
            # Extract features for time-step t
            X_t = X[:, t, :]  # (n, d)

            # Standardize features
            scaler = StandardScaler()
            X_t_scaled = scaler.fit_transform(X_t)
            self.scalers.append(scaler)

            # Initialize kernel machine (Kernel Ridge Regression here)
            model = KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma)
            self.models.append(model)

    def train(self):
        for t, model in enumerate(tqdm(self.models, desc="Training Kernel Machines")):
            # Extract and scale features for time-step t
            X_t = self.scalers[t].transform(self.X[:, t, :])
            y = self.expr.ravel()  # Flatten target
            model.fit(X_t, y)

    def predict(self, x, t: int):
        """
        Predict using the kernel machine for time-step t.
        Args:
            x: array-like or tensor of shape (n, m, d) or (n, d), input features
            t: int, time-step index
        Returns:
            predictions: array of shape (n, m) or (n,), predicted values
        """
        # print("Predicting with KernelEnsemble")
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        # print(f"Input x shape: {x.shape}")

        if x.ndim == 3:  # Case when x is (n, m, d)
            n, m, d = x.shape
            # print(f"Reshaping input x of shape (n, m, d): ({n}, {m}, {d})")
            x_reshaped = x.reshape(-1, d)  # Flatten to (n*m, d)
            # print(f"x_reshaped shape after flattening: {x_reshaped.shape}")
            x_scaled = self.scalers[t].transform(x_reshaped)  # Scale the flattened input
            # print(f"x_scaled shape after scaling: {x_scaled.shape}")
            output = self.models[t].predict(x_scaled).reshape(n, m)  # Predict and reshape back to (n, m)
            # print(f"Output shape after prediction and reshaping: {output.shape}")
        elif x.ndim == 2:  # Case when x is (n, d)
            # print(f"Scaling input x of shape (n, d): {x.shape}")
            x_t_scaled = self.scalers[t].transform(x)  # Scale directly
            # print(f"x_t_scaled shape after scaling: {x_t_scaled.shape}")
            output = self.models[t].predict(x_t_scaled)  # Predict directly
            # print(f"Output shape after prediction: {output.shape}")
        elif x.ndim == 1:  # Case when x is (d,)
            # print(f"Scaling input x of shape (d,): {x.shape}")
            x_t_scaled = self.scalers[t].transform(x[np.newaxis, :])
            # print(f"x_t_scaled shape after scaling: {x_t_scaled.shape}")
            output = self.models[t].predict(x_t_scaled)
        
        else:
            raise ValueError("Input x must have shape (n, m, d) or (n, d).")

        return output[...,  np.newaxis]

class KSVMEnsemble:
    """
    Ensemble of T independent SVM classifiers.
    Each SVM t learns to predict expr[:, 0] (either B or 0) from X[:, t, :].
    """
    def __init__(self,
                    X,                    # (n, T, d), array or Tensor
                    expr,                 # (n, 1), array or Tensor
                    B,                    # Reward value for positive class
                    kernel='rbf',
                    C=1.0,
                    gamma='scale'):
        if isinstance(X, torch.Tensor):
            self.X = X.cpu().numpy()
        else:
            self.X = X
        if isinstance(expr, torch.Tensor):
            self.expr = expr.cpu().numpy()
        else:
            self.expr = expr

        self.B = B
        self.T = X.shape[1]
        self.models = []
        self.scalers = []

        for t in range(self.T):
            # Extract features for time-step t
            X_t = self.X[:, t, :]  # (n, d)

            # Standardize features
            scaler = StandardScaler()
            X_t_scaled = scaler.fit_transform(X_t)
            self.scalers.append(scaler)

            # Initialize SVM classifier
            model = SVC(kernel=kernel, C=C, gamma=gamma)
            self.models.append(model)

    def train(self):
        for t, model in enumerate(tqdm(self.models, desc="Training SVM Models")):
            # Extract and scale features for time-step t
            X_t = self.scalers[t].transform(self.X[:, t, :])
            y = (self.expr.ravel() != 1).astype(int)  # Convert to binary labels (1 for B, 0 for 0)
            # print(f"Number of ones in y: {np.sum(y == 1)}")
            model.fit(X_t, y)
        print("Trained")

    def predict(self, x, t: int):
        """
        Predict using the SVM for time-step t.
        Args:
            x: array-like or tensor of shape (n, m, d) or (n, d), input features
            t: int, time-step index
        Returns:
            predictions: array of shape (n, m) or (n,), predicted rewards (B or 0)
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        if x.ndim == 3:  # Case when x is (n, m, d)
            n, m, d = x.shape
            x_reshaped = x.reshape(-1, d)  # Flatten to (n*m, d)
            x_scaled = self.scalers[t].transform(x_reshaped)  # Scale the flattened input
            output = self.models[t].predict(x_scaled).reshape(n, m)  # Predict and reshape back to (n, m)
        elif x.ndim == 2:  # Case when x is (n, d)
            x_scaled = self.scalers[t].transform(x)  # Scale directly
            output = self.models[t].predict(x_scaled)  # Predict directly
        else:
            raise ValueError("Input x must have shape (n, m, d) or (n, d).")
        return np.exp(output * self.B)[..., np.newaxis]  # Convert binary labels back to rewards (B or 0)

class RewardEstimator(nn.Module):
    """
    A single neural network to estimate rewards for t = T.
    """
    def __init__(self,
                 X,                    # (n, T, d), array or Tensor
                 expr,                 # (n, 1), array or Tensor
                 hidden_sizes=(64, 64),
                 batch_size=128,
                 lr=1e-3,
                 device=None):
        super().__init__()

        # 1) DEVICE
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
                                else torch.device("cpu")
        )
        if isinstance(X, np.ndarray):
            X = X.copy()
            X = torch.from_numpy(X).float()
        if isinstance(expr, np.ndarray):
            expr = expr.copy()
            expr = torch.from_numpy(expr).float()

        n, T, d = X.shape
        y = expr.view(-1)   # (n,)

        # 2) SELECT LAST TIME-STEP (t = T)
        X_T = X[:, -1, :].contiguous()  # (n, d)

        # 3) BUILD DATASET & DATALOADER
        self.dataset = TensorDataset(X_T, y)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # 4) BUILD MLP
        layers = []
        in_dim = d
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h),
                       nn.ReLU(inplace=True)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers).to(self.device)

        # 5) OPTIMIZER & LOSS
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, epochs: int = 20, verbose: bool = False):
        for epoch in range(1, epochs + 1):
            self.net.train()
            total_loss = 0.0
            for x_batch, y_batch in self.loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred = self.net(x_batch).squeeze(-1)
                loss = self.criterion(pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * x_batch.size(0)

            avg_mse = total_loss / len(self.loader.dataset)
            if verbose:
                print(f"[Epoch {epoch:02d}] avg MSE = {avg_mse:.4f}")

    def predict(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)

        self.net.eval()
        with torch.no_grad():
            predictions = self.net.forward(x)        # Shape (n,)
        return predictions.numpy()
