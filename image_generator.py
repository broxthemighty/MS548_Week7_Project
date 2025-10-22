# image_generator.py
from diffusers import StableDiffusionPipeline
import torch
import os

# Initialize model once to save memory
_model_cache = {}

def load_model(model_name="runwayml/stable-diffusion-v1-5", device="cuda"):
    """Load and cache models by name."""
    if model_name not in _model_cache:
        print(f"Loading {model_name}...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        _model_cache[model_name] = pipe
    return _model_cache[model_name]

def generate_image(
    prompt,
    output_dir="generated_images",
    model_name="runwayml/stable-diffusion-v1-5",
    steps=20,
    guidance=7.5,
    size=(512, 512),
    device="cuda"
):
    """Generate an image using the chosen model and settings."""
    os.makedirs(output_dir, exist_ok=True)
    pipe = load_model(model_name, device)

    print(f"Generating image for: {prompt}")
    result = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]

    result = result.resize(size)
    output_path = os.path.join(output_dir, "generated_image.png")
    result.save(output_path)
    return output_path
