import torch

# --- load checkpoint ---
ckpt_path = "_work-exp-01/_work/checkpoints/last.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu")

# if it's a Lightning checkpoint, weights are under "state_dict"
state_dict = ckpt.get("state_dict", ckpt)

# --- iterate through all parameters ---
for name, tensor in state_dict.items():
    if not torch.is_tensor(tensor):
        continue
    arr = tensor.float()  # ensure float32 for stats
    print(
        f"{name:40s} "
        #f"shape={list(arr.shape):30s} "
        f"max={arr.max().item():.5f} "
        f"min={arr.min().item():.5f} "
        f"mean={arr.mean().item():.5f} "
        f"std={arr.std().item():.5f}"
    )
