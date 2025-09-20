import torch

@torch.no_grad()
def quantize(w: torch.Tensor, axis=None, dtype: str = "uint8", qtype: str = "linear"):
    if dtype == "uint8":
        qmin, qmax = 0.0, 255.0
    elif dtype == "uint4":
        qmin, qmax = 0.0, 15.0
    else:
        raise NotImplementedError("Only uint8 and uint4 are currently supported")

    w = w.float()
    sign = None
    if qtype == "nonlinear":
        sign = torch.sign(w)
        sign[sign == 0] = 1.0
        x = torch.log(torch.abs(w) + 1e-12)
    elif qtype == "linear":
        x = w
    else:
        raise ValueError("qtype must be 'linear' or 'nonlinear'")

    if axis is None:
        xmin, xmax = x.min(), x.max()
        scale = (xmax - xmin) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-12)
        zp = (-xmin / scale).round().clamp(qmin, qmax)
        q = (x / scale + zp).round().clamp(qmin, qmax).to(torch.uint8)
        return q, scale, zp.to(torch.uint8), None, sign
    else:
        dims = [d for d in range(w.ndim) if d != axis]
        xmin = x.amin(dims, keepdim=True)
        xmax = x.amax(dims, keepdim=True)
        scale = (xmax - xmin) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-12)
        zp = (-xmin / scale).round().clamp(qmin, qmax)
        q = (x / scale + zp).round().clamp(qmin, qmax).to(torch.uint8)
        return q, scale.squeeze().reshape(-1), zp.squeeze().reshape(-1).to(torch.uint8), axis, sign

@torch.no_grad()
def dequantize(q: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor, axis=None, qtype: str = "linear", sign: torch.Tensor = None):
    if axis is None:
        x = (q.float() - zp.float()) * scale.float()
    else:
        shape = [1] * q.ndim
        shape[axis] = q.shape[axis]
        x = (q.float() - zp.view(*shape).float()) * scale.view(*shape).float()
    if qtype == "nonlinear":
        if sign is None:
            raise ValueError("sign is required for nonlinear dequantization")
        mag = torch.exp(x) - 1e-12
        return sign.float() * mag
    elif qtype == "linear":
        return x
    else:
        raise ValueError("qtype must be 'linear' or 'nonlinear'")