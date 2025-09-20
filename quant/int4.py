import torch

def pack_uint4(q: torch.Tensor) -> torch.Tensor:
    """Packs uint8 tensor with values [0...15] to uint8 tensor of length // 2"""
    q = q.to(torch.uint8).view(-1)
    if q.numel() % 2 == 1:
        q = torch.cat([q, torch.zeros(1, dtype=torch.uint8, device=q.device)], dim=0)
    low = q[0::2] & 15
    high = (q[1::2] & 15) << 4
    return (low | high).to(torch.uint8)

def unpack_uint4(packed: torch.Tensor, total_elems: int) -> torch.Tensor:
    """Unpackes packed tensor back to uint8"""
    p = packed.to(torch.uint8).view(-1)
    low = p & 15
    high = (p >> 4) & 15
    out = torch.empty(p.numel() * 2, dtype=torch.uint8, device=p.device)
    out[0::2] = low
    out[1::2] = high
    return out[:total_elems]