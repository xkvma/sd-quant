import torch

@torch.no_grad()
def pack_sign_bits(sign: torch.Tensor) -> torch.Tensor:
    b = (sign > 0).to(torch.uint8).flatten()
    pad = (-b.numel()) % 8
    if pad:
        b = torch.cat((b, torch.zeros(pad, dtype=torch.uint8, device=b.device)))
    b = b.view(-1, 8)
    return (b[:,0] | (b[:,1] << 1) | (b[:,2] << 2) | (b[:,3] << 3) |
            (b[:,4] << 4) | (b[:,5] << 5) | (b[:,6] << 6) | (b[:,7] << 7)).contiguous()

@torch.no_grad()
def unpack_sign_bits(packed: torch.Tensor, total_elems: int, device=None) -> torch.Tensor:
    device = packed.device if device is None else device
    x = packed.to(torch.uint8).view(-1, 1)
    bits = torch.cat([ (x >> i) & 1 for i in range(8) ], dim=1).flatten()[:total_elems]
    return torch.where(bits.bool(),
                       torch.tensor(1.0, device=device),
                       torch.tensor(-1.0, device=device)).view(-1)