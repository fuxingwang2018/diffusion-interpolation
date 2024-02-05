import torch 


def _pack_xy(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B, 2, C, H, W) or (2, C, H, W)
    y: (B, K, C, H, W) or (K, C, H, W)
    returns:
        cond:   (B, 2*C, H, W)
        target: (B, K*C, H, W)
    """
    if x.dim() == 4:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    B, two, C, H, W = x.shape
    _, K, C2, H2, W2 = y.shape
    if not (two == 2 and C2 == C and H2 == H and W2 == W):
        raise ValueError(f"Mismatch in (x,y) shapes: x={tuple(x.shape)} y={tuple(y.shape)}")
    cond = x.reshape(B, 2 * C, H, W)
    target = y.reshape(B, K * C, H, W)
    return cond, target