import os
import torch
import torch.nn as nn
from .qf import quantize, dequantize
from .bits import pack_sign_bits, unpack_sign_bits
from .int4 import pack_uint4, unpack_uint4

@torch.no_grad()
def quantize_model(model: nn.Module, qtype: str, dtype: str, save_path: str):
    assert qtype in ("linear", "nonlinear")
    assert dtype in ("uint8", "uint4")

    qstate = {}
    for name, m in model.named_modules():
        if not hasattr(m, "weight") or m.weight is None:
            continue

        W = m.weight.data
        axis = 0 if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)) else None

        q, scale, zp, used_axis, sign = quantize(W, axis=axis, dtype=dtype, qtype=qtype)

        entry = {
            "qtype": qtype,
            "dtype": dtype,
            "scale": scale.cpu(),
            "zp": zp.cpu(),
            "axis": used_axis,
        }

        if dtype == "uint4":
            entry["packed_q"] = pack_uint4(q).cpu()
            entry["orig_shape"] = torch.tensor(list(q.shape), dtype=torch.int64)
            entry["total_elems"] = torch.tensor(q.numel(), dtype=torch.int64)
        else:
            entry["q"] = q.cpu()

        if qtype == "nonlinear":
            packed_sign = pack_sign_bits(sign.cpu().to(torch.int8))
            entry["packed_sign"] = packed_sign
            entry["total_elems_sign"] = torch.tensor(sign.numel(), dtype=torch.int64)

        if hasattr(m, "bias") and m.bias is not None:
            entry["bias_fp32"] = m.bias.data.cpu()

        qstate[name] = entry

    torch.save({"layers": qstate, "dtype": dtype, "qtype": qtype}, save_path)
    return save_path

@torch.no_grad()
def load_quantized_weights(model: nn.Module, weights_path: str):
    ckpt = torch.load(weights_path, map_location="cpu")
    layers = ckpt["layers"]
    name_to_mod = {n: m for n, m in model.named_modules()}
    for name, e in layers.items():
        if name not in name_to_mod:
            continue
        m = name_to_mod[name]
        device = m.weight.device
        axis = e["axis"]
        dtype = e.get("dtype", "uint8")
        qtype = e.get("qtype", "linear")

        if dtype == "uint4":
            total = int(e["total_elems"])
            q = unpack_uint4(e["packed_q"], total).to(device)
            q = q.view(*e["orig_shape"].tolist())
        else:
            q = e["q"].to(device)

        scale = e["scale"].to(device)
        zp = e["zp"].to(device)
        sign = None
        
        if qtype == "nonlinear":
            total_sign = int(e["total_elems_sign"])
            sign = unpack_sign_bits(e["packed_sign"], total_sign, device=device)
            sign = sign.view_as(q)
        w = dequantize(q, scale, zp, axis=axis, sign=sign, qtype=qtype)

        m.weight.data.copy_(w.to(dtype=m.weight.dtype))
        if "bias_fp32" in e and getattr(m, "bias", None) is not None:
            m.bias.data.copy_(e["bias_fp32"].to(device=device, dtype=m.bias.dtype))