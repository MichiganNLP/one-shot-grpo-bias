import torch

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt

def compute_l2_update(ckpt1_path, ckpt2_path):
    sd1 = load_checkpoint(ckpt1_path)
    sd2 = load_checkpoint(ckpt2_path)

    total_norm = 0.0

    for k in sd1:
        if k not in sd2:
            continue
        diff = sd2[k] - sd1[k]
        total_norm += torch.sum(diff.float() ** 2).item()

    return total_norm ** 0.5


if __name__ == "__main__":
    ckpt1 = "checkpoint_step_0.pt"
    ckpt2 = "checkpoint_step_100.pt"

    update_norm = compute_l2_update(ckpt1, ckpt2)
    print(f"L2 parameter update norm: {update_norm}")