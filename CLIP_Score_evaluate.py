import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

# 設定裝置
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 圖片與 prompt 資料夾
image_dir = "Image_Text_to_Image_2"
results = []

# 最多 50 張圖
for idx in tqdm(range(50), desc="Evaluating CLIP scores"):
    image_path = os.path.join(image_dir, f"gta5_{idx:03d}.png")
    prompt_path = os.path.join(image_dir, f"gta5_{idx:03d}.txt")

    if not os.path.exists(image_path) or not os.path.exists(prompt_path):
        continue

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    score = outputs.logits_per_image[0][0].item()

    results.append({
        "image": os.path.basename(image_path),
        "prompt": prompt,
        "clip_score": score
    })

# 輸出成 CSV
df = pd.DataFrame(results).sort_values(by="clip_score", ascending=False)
df.to_csv("clip_results_2.csv", index=False, encoding="utf-8")
print("✅ Done! Results saved to clip_results_2.csv")

df = pd.read_csv("clip_results_2.csv")

plt.figure(figsize=(14, 6))
plt.plot(df["image"], df["clip_score"], marker='o', linestyle='-', color='blue')
plt.xticks(rotation=90)
plt.xlabel("Image Filename")
plt.ylabel("CLIP Score")
plt.title("Top 50 CLIP Scores for GTA5-style Images")
plt.grid(True)
plt.tight_layout()

# 儲存圖表為 PNG 檔
plt.savefig("clip_results_2.png")
plt.show()
