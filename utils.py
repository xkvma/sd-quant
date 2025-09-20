import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchvision import transforms
from transformers import PreTrainedTokenizer, PreTrainedModel, CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, SchedulerMixin, PNDMScheduler

def visualize_grid(images, n_cols=2, figsize_width=8):
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