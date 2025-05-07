import os
import torch
import zipfile
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# 1. 建立資料夾
save_dir = "image_Text_to_Image_2"
os.makedirs(save_dir, exist_ok=True)

# 2. 讀取高質量prompts.txt
with open("prompts.txt", "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

print(f"✅ Loaded {len(prompts)} prompts.")

# 3. 載入模型
pipe = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker=None,
).to("cuda")


# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

# 4. 生成圖片
for idx, prompt in enumerate(tqdm(prompts, desc="Generating Pro Images")):
    image = pipe(
        prompt,
        height=768,
        width=768,
        num_inference_steps=50,  # ✅ 增加到50步
        guidance_scale=10.0      # ✅ guidance強化
    ).images[0]

    # 儲存圖片
    img_save_path = os.path.join(save_dir, f"gta5_{idx:03d}.png")
    image.save(img_save_path)

    # 儲存prompt
    txt_save_path = os.path.join(save_dir, f"gta5_{idx:03d}.txt")
    with open(txt_save_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"Saved {img_save_path} and {txt_save_path}")

# 5. 打包成ZIP
zip_filename = "Image_Text_to_Image_2.zip"
with zipfile.ZipFile(zip_filename, "w") as zipf:
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, save_dir)
            zipf.write(file_path, arcname)

print(f"✅ All done! Dataset zipped to {zip_filename}")
