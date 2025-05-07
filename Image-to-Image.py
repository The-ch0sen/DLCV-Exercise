import os
import random
from tqdm import tqdm
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline

# 1. 設定資料夾
input_folder = "gta5_in_images"  # 放60張原圖
output_folder = "gta5_generated_images"
os.makedirs(output_folder, exist_ok=True)

# 2. 載入模型
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "dreamlike-art/dreamlike-photoreal-2.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker=None
).to("cuda")
pipe.enable_attention_slicing()

# 3. 設定 prompt 基本結構
base_prompt = "cinematic urban street scene, highly detailed, realistic, in the style of Grand Theft Auto V"
prompt_variations = [
    " at night", " under sunset", " during rain", " in heavy fog", " busy downtown",
    " abandoned street", " traffic jam", " neon lights", " early morning light", " stormy sky"
]

# 4. 處理每張圖
valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
input_images = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
total_target_images = 1000
image_counter = 0

print(f"✅ Found {len(input_images)} input images.")

while image_counter < total_target_images:
    for img_name in input_images:
        if image_counter >= total_target_images:
            break
        
        img_path = os.path.join(input_folder, img_name)
        init_image = Image.open(img_path).convert("RGB").resize((768,768))

        # 隨機強度與prompt
        strength = random.uniform(0.4, 0.7)
        guidance_scale = random.uniform(7.5, 9.0)
        extra_prompt = random.choice(prompt_variations)
        prompt = base_prompt + extra_prompt

        output = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=100
        )

        output_img = output.images[0]

        save_path = os.path.join(output_folder, f"gta5_augmented_{image_counter:04d}.png")
        output_img.save(save_path)

        print(f"✅ Saved {save_path} | strength={strength:.2f} | guidance={guidance_scale:.2f}")

        image_counter += 1

print("✅ Done! Generated all 1000 images.")
