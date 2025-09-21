# sd-quant
Stable diffusion quantized inference from scratch

![example](example.png "fp32 -> uint4")

The example.ipynb notebook demonstrates Stable Diffusion inference built from modular parts and implements linear and log quantization to uint8/uint4.

### Usage 

```bash
python3 -m venv .venv
. .venv/bin/activate 
pip install -r requirements.txt
```

Then run example.ipynb notebook