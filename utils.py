import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchvision import transforms
from transformers import PreTrainedTokenizer, PreTrainedModel, CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, SchedulerMixin, PNDMScheduler

def visualize_grid(images, n_cols=2, figsize_width=6):
    num_images = len(images)
    n_rows = (num_images + n_cols - 1) // n_cols 
    figsize_height = figsize_width * n_rows / n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_width, figsize_height))

    for i, img in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    
def load_models(model_id):
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")

    return {
          "tokenizer": tokenizer,
          "text_encoder": text_encoder,
          "unet": unet,
          "scheduler": scheduler,
          "vae": vae,
        }

def diffs(a,b):
    d=(a.float()-b.float()).abs()
    return dict(mae=d.mean().item(), rmse=(d.pow(2).mean().sqrt()).item(),
                max=d.max().item(), rel=(d/(a.abs()+1e-12)).mean().item())

def model_size_bytes(state_dict):
    s=0
    for t in state_dict.values():
        if isinstance(t, torch.Tensor):
            s += t.numel() * t.element_size()
        elif isinstance(t, dict):
            for t2 in t.values():
                if isinstance(t2, torch.Tensor):
                    s += t2.numel() * t2.element_size()
    return s