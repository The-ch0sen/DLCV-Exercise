import os
import shutil
from PIL import Image

# === 資料夾設定 ===
input_folder = "gta5_in_images"
output_folder = "gta5_input_images_only"

# === 清理舊資料夾並建立新資料夾 ===
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# === 支援的圖片格式 ===
supported_exts = ('.tiff', '.tif', '.png', '.jpg', '.jpeg')

# === 圖片轉換處理 ===
count = 0
for file in os.listdir(input_folder):
    ext = file.lower()
    if ext.endswith(supported_exts):
        src = os.path.join(input_folder, file)
        dst = os.path.join(output_folder, os.path.splitext(file)[0] + ".jpg")
        try:
            img = Image.open(src).convert("RGB")
            img.save(dst, "JPEG", quality=95)
            count += 1
        except Exception as e:
            print(f"❌ Failed to convert {file}: {e}")

print(f"✅ Converted {count} images to JPG in '{output_folder}'")
