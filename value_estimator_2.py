import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
        time_feat = torch.tensor([t / (self.T - 1)], # this was divided by T - 1 before
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
    
    def predict(self, x, t):
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

        # print(f"x shape: {x.shape}, t shape: {t.shape}")
        # Build time features and concatenate with x
        time_feat = t.unsqueeze(-1) / (self.T - 1)  # Normalize t #this had a -1 before
        inp = torch.cat([x, time_feat], dim=-1)    # Concatenate x and t

        # Predict
        self.net.eval()
        with torch.no_grad():
            predictions = self.forward(inp)        # Shape (n,)
        return predictions.numpy()