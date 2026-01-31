# ColorizeNet
# Image Colorization Project (CNN → GAN)

## 📌 Project Overview

This project focuses on **automatic image colorization** using Deep Learning.

Given a **grayscale image**, the model predicts **realistic color information** and reconstructs a full-color image.  
We start with a **CNN-based U-Net model** and later upgrade to a **GAN-based (Pix2Pix) approach** for improved realism.

This repository is structured so that **any team member can clearly understand the full pipeline**, from data loading to final image generation.

---

## 🧠 Problem Statement

- **Input:** Grayscale image  
- **Output:** Colorized RGB image  
- **Learning Type:** Supervised learning (image-to-image regression)

This is **NOT a classification problem**.  
The model predicts **color values per pixel**, not labels.

---

## 🧪 Dataset

### Dataset Used
- **Place365 dataset** (or any large natural image dataset)

### Important Notes
- Scene class labels are **NOT required**
- Only image files are used
- Dataset is treated as **unlabeled images**

---

## 🎯 Core Idea (LAB Color Space)

We use **LAB color space** instead of RGB.

- **L channel:** Lightness (grayscale structure)
- **a, b channels:** Color information

### Why LAB?
- Separates structure from color
- Makes learning color prediction easier
- Industry standard for colorization tasks

---

## 🔄 Complete Pipeline (Step-by-Step)

### 1️⃣ Load Images
- Load RGB images from the dataset directory.

---

### 2️⃣ Resize Images
- Resize all images to a fixed size (e.g., `128×128` or `256×256`)
- Required for batching and CNN input consistency

---

### 3️⃣ Convert RGB → LAB
- Convert each image from RGB to LAB color space.

---

### 4️⃣ Split Channels
- **Input (X):** L channel  
  - Shape: `(H, W, 1)`
- **Target (Y):** a and b channels  
  - Shape: `(H, W, 2)`

---

### 5️⃣ Normalize Data

Normalization is mandatory for stable training.

- Normalize **L channel**:
L = L / 100


- Normalize **a and b channels**:
a = a / 128
b = b / 128


---

### 6️⃣ Dataset Splitting
- If dataset already has splits → use them
- Otherwise:
- 80% Training
- 10% Validation
- 10% Testing

Validation data is required for:
- Early stopping
- Hyperparameter tuning

---

### 7️⃣ Batch Creation
- Use batching for efficient GPU training
- Typical batch sizes:
- 16 (safe)
- 32 (standard)

---

## 🏗️ Model Architecture (Baseline)

### CNN-Based Model
- **Architecture:** Encoder–Decoder CNN
- **Recommended:** **U-Net**

### Model Details
- Input: `(H, W, 1)` → L channel
- Output: `(H, W, 2)` → a,b channels

U-Net preserves spatial details and edges, which is critical for colorization.

---

## 📉 Loss Function

This is a **regression task**, not classification.

- **Primary loss:**  
- MAE (L1 loss) or  
- MSE (L2 loss)

Accuracy metrics are **not applicable**.

---

## ⏱️ Training Utilities

### Callbacks Used
- EarlyStopping (monitor validation loss)
- ModelCheckpoint (save best model)
- ReduceLROnPlateau (optional)

---

## 🚀 Training Phase

- Train the model using:
- Input: L channel images
- Target: a,b channels
- Monitor validation loss to prevent overfitting

---

## 🎨 Inference: Generating New Colorized Images

This is how new images are colorized after training.

### Step-by-Step Inference

1. Load a new RGB image
2. Convert RGB → LAB
3. Extract and normalize L channel
4. Pass L channel to trained model
5. Model predicts a,b channels
6. Denormalize predicted a,b
7. Combine original L + predicted a,b
8. Convert LAB → RGB
9. Output final colorized image

Only the **trained model** is used during inference.

---

## 🤖 GAN-Based Colorization (Advanced Stage)

After the CNN baseline works, we move to GANs.

### GAN Type Used
- **Conditional GAN (Pix2Pix)**

---

### GAN Architecture

#### Generator
- U-Net
- Input: L channel
- Output: a,b channels

#### Discriminator
- PatchGAN
- Input: (L + a,b)
- Output: Real / Fake decision

The discriminator checks whether the colors look realistic **for a given grayscale image**.

---

## 📊 GAN Loss Functions

### Generator Loss
Combination of:
- Adversarial loss (fool discriminator)
- L1 loss (stay close to ground truth)

Total Generator Loss = GAN Loss + λ × L1 Loss


---

### Discriminator Loss
- Binary cross-entropy
  - Real images → 1
  - Fake images → 0

---

## 🔁 GAN Training Loop

For each batch:
1. Train Discriminator on real and fake pairs
2. Freeze Discriminator
3. Train Generator to fool Discriminator
4. Repeat

---

## 🎯 GAN Inference

- Only the **Generator** is used
- Discriminator is discarded
- Inference steps are same as CNN-based colorization

---

## 📈 Expected Improvements with GANs

- Sharper images
- More realistic colors
- Better texture preservation
- Less color averaging

---

## 🧭 Project Roadmap

1. CNN + U-Net + MAE (baseline)
2. Improve preprocessing & augmentation
3. Add perceptual loss
4. Pix2Pix GAN implementation
5. Advanced GAN stabilization
6. Diffusion-based colorization (future)

---

## 👥 Team Notes

- This README defines the **single source of truth** for the project
- Any changes to pipeline or architecture should be reflected here
- All teammates should follow the same preprocessing and normalization steps

---
