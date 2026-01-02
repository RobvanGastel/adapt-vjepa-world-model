import torch

def pca(X : torch.Tensor, n_components : int = 3) -> torch.Tensor:
    Z_mean = X.mean(0, keepdim=True)
    Z = X - Z_mean
    U, S, VT = torch.linalg.svd(Z, full_matrices=False)
    
    max_col = torch.argmax(torch.abs(U), dim=0)
    signs = torch.sign(U[max_col, range(U.shape[1])])
    VT *= signs[:, None]

    Z = torch.matmul(Z, VT[:n_components].T)
    return Z

def min_max(
        X : torch.Tensor,
        target_min : float = 0.0,
        target_max : float = 1.0
    ) -> torch.Tensor:
    eps = 1e-8
    X_std = (X - X.min(0, True).values) / (X.max(0, True).values - X.min(0, True).values + eps)
    X_scaled = X_std * (target_max - target_min) + target_min
    return X_scaled