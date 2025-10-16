
import os, glob, pickle, gzip, json
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def save_samples(name: str, arr):
    import numpy as np, torch
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 1: arr = arr[:, None]
    path = f"/mnt/data/{name}.npy"
    np.save(path, arr)
    return path

def load_any(path: str):
    import numpy as np, pandas as pd, pickle, gzip, os
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 1: arr = arr[:, None]
        return arr
    if ext == ".npz":
        npz = np.load(path)
        key = max(npz.files, key=lambda k: npz[k].size)
        arr = npz[key]
        if arr.ndim == 1: arr = arr[:, None]
        return arr
    if ext == ".csv":
        import pandas as pd
        arr = pd.read_csv(path).values
        if arr.ndim == 1: arr = arr[:, None]
        return arr
    if ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            best = None
            for v in obj.values():
                if isinstance(v, np.ndarray) and (best is None or v.size > best.size):
                    best = v
            if best is None:
                raise ValueError("Pickle has no ndarray")
            arr = best
        elif isinstance(obj, np.ndarray):
            arr = obj
        else:
            raise ValueError("Unsupported pickle payload")
        if arr.ndim == 1: arr = arr[:, None]
        return arr
    if ext == ".gz":
        with gzip.open(path, "rb") as f:
            arr = np.load(f)
            if arr.ndim == 1: arr = arr[:, None]
            return arr
    raise ValueError(f"Unsupported: {path}")

def load_glob(pattern: str):
    paths = sorted(glob.glob(pattern))
    arrays = [load_any(p) for p in paths]
    return paths, arrays

def tv_bounds_classifier(X_star: np.ndarray,
                         X_hat: np.ndarray,
                         max_samples_star: int = 5000,
                         max_samples_hat: int = 50000,
                         random_state: int = 42) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)
    if X_star.shape[0] > max_samples_star:
        idx = rng.choice(X_star.shape[0], size=max_samples_star, replace=False)
        X_star = X_star[idx]
    if X_hat.shape[0] > max_samples_hat:
        idx = rng.choice(X_hat.shape[0], size=max_samples_hat, replace=False)
        X_hat = X_hat[idx]

    n_star, d_star = X_star.shape[0], X_star.shape[1]
    n_hat, d_hat = X_hat.shape[0], X_hat.shape[1]
    if d_star != d_hat:
        raise ValueError(f"Dim mismatch: p_star d={d_star}, p_hat d={d_hat}")
    d = d_star

    X = np.vstack([X_star, X_hat])
    y = np.hstack([np.ones(n_star), np.zeros(n_hat)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)
    )
    clf.fit(X_train, y_train)

    p_test = clf.predict_proba(X_test)[:, 1]
    y_pred = (p_test >= 0.5).astype(int)
    test_err = 1.0 - accuracy_score(y_test, y_pred)
    tv_lb = max(0.0, 1.0 - 2.0 * test_err)
    tv_plugin = float(np.mean(np.abs(2.0 * p_test - 1.0)))

    p_star_hat = clf.predict_proba(X_star)[:, 1]
    p_hat_hat  = clf.predict_proba(X_hat)[:, 1]
    eps = 1e-8
    p_star_hat = np.clip(p_star_hat, eps, 1 - eps)
    p_hat_hat  = np.clip(p_hat_hat,  eps, 1 - eps)
    r_star = p_star_hat / (1.0 - p_star_hat)
    r_hat  = p_hat_hat  / (1.0 - p_hat_hat)
    kl_star_hat = float(np.mean(np.log(r_star)))
    kl_hat_star = float(np.mean(-np.log(r_hat)))
    tv_ub = float(min(np.sqrt(0.5 * max(kl_star_hat, 0.0)),
                      np.sqrt(0.5 * max(kl_hat_star, 0.0))))

    return dict(
        n_star=n_star, n_hat=n_hat, d=d,
        test_error=float(test_err),
        tv_lower_bound=float(tv_lb),
        tv_plugin=float(tv_plugin),
        kl_star_hat=float(kl_star_hat),
        kl_hat_star=float(kl_hat_star),
        tv_upper_bound=float(tv_ub),
    )
