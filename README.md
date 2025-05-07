# 🧠 DLCV Exercise 1 – Diffusion Model (Text-to-Image & Image-to-Image)

## 🎯 Objective

This project demonstrates the ability to use **diffusion models** for both:
- Text-to-Image generation (using prompts in GTA5 style)
- Image-to-Image generation (style transfer from real GTA5 dataset images)

We also evaluate the results using **CLIP Score** and **FID**.

---

## 📁 Folder Structure

```bash
.
├── Text-to-Image_1.py           # dreamlike-art model
├── Text-to-Image_2.py           # Realistic Vision model
├── prompts.txt                  # 50 GTA5-style prompts
├── CLIP_Score_evaluate.py       # Evaluate CLIP score
├── Image-to-Image.py            # Image-to-Image generation
├── FID_evaluate.py              # FID evaluation using TorchMetrics
├── gta5_in_images/              # Original images (.tiff from GTA5 dataset)
├── gta5_generated_images/       # Generated 1000 images
├── gta5_input_images_only/      # Converted input images (.jpg)
├── clip_results_1.csv/png       # CLIP results for Model 1
├── clip_results_2.csv/png       # CLIP results for Model 2
└── evaluation_plot.png          # FID/KID chart
```

---

## 🖼️ Topic 1 – Text-to-Image (T2I)

### ✅ Models Used:
- [`dreamlike-art/dreamlike-photoreal-2.0`](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0)
- [`SG161222/Realistic_Vision_V5.1_novae`](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVA)

### ✅ Pipeline
1. Load prompts from `prompts.txt` (GTA5-style text descriptions)
2. Use Hugging Face `StableDiffusionPipeline` for inference
3. Generate 50 images per model
4. Save image + text pairs in respective folders
5. Use `CLIP_Score_evaluate.py` to compute CLIP Score (semantic similarity)
6. Results are saved as `.csv` and plotted using `matplotlib`

### 📊 CLIP Score Comparison
- **Realistic Vision** gave more stable and semantically consistent results
- **Dreamlike Photoreal** had larger score fluctuations and inconsistency in later samples

---

## 🖼️ Topic 2 – Image-to-Image (I2I)

### ✅ Dataset
- From [Kaggle - GTA5 Dataset](https://www.kaggle.com/datasets/joebeachcapital/grand-theft-auto-5)
- 60 `.tiff` images converted to `.jpg` using `Image_transfer.py`

### ✅ Pipeline
1. Use `dreamlike-art` model for image transformation
2. Each image gets a prompt variation (e.g., "at night", "in fog")
3. Generate 1000 images total using only 60 original images
4. Store results in `gta5_generated_images/`

---

## 📏 Evaluation Metrics

### ✅ CLIP Score (`clip-vit-base-patch32`)
- Measures how well the generated image matches the prompt
- Used only for T2I

### ✅ FID (Fréchet Inception Distance)
- Measures distributional similarity between real and fake images
- Used for I2I comparison (60 real vs 1000 generated)
- Implemented via `torchmetrics.image.FrechetInceptionDistance`

---

## 🔧 Environment Setup

```bash
pip install diffusers transformers torch torchvision tqdm torchmetrics pandas matplotlib
```

> 建議使用虛擬環境以避免套件衝突（如 numpy binary incompatibility）

---

## 📬 Contact

For any questions, feel free to reach out.  
黃昱豪 @ National Chung Cheng University（中正大學）
