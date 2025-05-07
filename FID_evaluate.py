import os
from PIL import Image
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

# === 路徑設定 ===
real_dir = "gta5_input_images_only"
fake_dir = "gta5_generated_images"

# === 預處理：Resize + Normalize (to [0,1]) ===
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

# === 載入圖片 ===
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(os.path.join(folder, filename)).convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor)
    return torch.stack(images)

# === 載入資料 ===
print("📥 Loading real images...")
real_images = load_images_from_folder(real_dir)
print(f"✅ Loaded {len(real_images)} real images")

print("📥 Loading generated images...")
fake_images = load_images_from_folder(fake_dir)
print(f"✅ Loaded {len(fake_images)} fake images")

# === 計算 FID ===
fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)
fid_score = float(fid.compute())

print(f"\n📊 FID Score (TorchMetrics): {fid_score:.2f}")
