import torch
from tqdm import tqdm
from torchvision import transforms
from transformers import PreTrainedTokenizer, PreTrainedModel
from diffusers import UNet2DConditionModel, AutoencoderKL, SchedulerMixin
from PIL import Image
from typing import List

def sd_inference(
    tokenizer: PreTrainedTokenizer,
    text_encoder: PreTrainedModel,
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    vae: AutoencoderKL,
    text_prompt: str,
    guidance_scale = 7.5,
    num_images = 4,
    device="cuda",
) -> List[Image.Image]:
    with torch.no_grad():
        # Tokenize prompt
        inputs = tokenizer([text_prompt] * num_images, return_tensors="pt").to(device)

        # Encode prompt to embedding
        text_encoder.to(device)
        text_embeddings = text_encoder(**inputs).last_hidden_state.to(device)
        
        max_length = inputs.input_ids.shape[-1]
        uncond_input = tokenizer([""] * num_images, padding="max_length", max_length=max_length, return_tensors="pt").to(device)
        uncond_embeddings = text_encoder(**uncond_input).last_hidden_state.to(device)

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)
        text_encoder.to("cpu")

        # Generate random gaussian noise
        latents = torch.randn((num_images, vae.config.latent_channels, 64, 64)).to(device)
        latents = latents * scheduler.init_noise_sigma

        # Apply diffusion steps
        unet.to(device)
        for t in tqdm(scheduler.timesteps):
            
            latent_model_input = scheduler.scale_model_input(latents, t)
            latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)

            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # A weighted sum of conditional and unconditional embeddings to regulate prompt's influence
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample
        unet.to("cpu")

        # Denormalize latents
        latents = 1 / vae.config.scaling_factor * latents

        # Project latents to image space
        vae.to(device)
        images = vae.decode(latents).sample.cpu()
        vae.to("cpu")
        del latents

        # Denormalize and convert to PIL
        images = (images / 2 + 0.5).clamp(0, 1)
        to_pil = transforms.ToPILImage()
        final_images = [to_pil(img) for img in images]

        return final_images