# prodigy_image_generation.py

from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt: str, filename: str = "output.png"):
    try:
        # Load Stable Diffusion model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Loading model on {device.upper()}...")

        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        pipe = pipe.to(device)

        # Generate image
        print(f"🎨 Generating image for: '{prompt}'")
        image = pipe(prompt).images[0]

        # Save result
        image.save(filename)
        print(f"✅ Image saved as '{filename}'")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧠 Image Generation with Stable Diffusion v1.5")
    prompt = input("🖼️ Enter your image prompt: ")
    generate_image(prompt)
