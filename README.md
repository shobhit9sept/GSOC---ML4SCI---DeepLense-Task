# GSOC 2025 - ML4SCI - DeepLense Assignment

This document serves as my submission for the DeepLense assignment as part of GSOC 2025. I am particularly interested in the **"Foundation Models in Gravitational Lensing"** project under DeepLense and have completed the following tasks accordingly.

---

## Task I: Image Classification using PyTorch/Keras

### Approach
I fine-tuned **ViT-base-patch16-224** for the strong lensing image classification task using the provided dataset, which contains **10K images per class**.

### Training Details
- **Optimizer:** AdamW (lr=5e-5, weight_decay=0.05)
- **Learning Rate Scheduler:** CosineAnnealingLR (T_max=10)
- **Epochs:** 15
- **Performance Metrics:**
  - **Training Accuracy:** 89.34%
  - **Validation Accuracy:** 88.63%
  - **Micro-average AUC:** 0.9648
  - **Class-wise AUC-ROC Scores:**
    - **Class No:** 0.9767
    - **Class Sphere:** 0.9463
    - **Class Vort:** 0.9692
  - **Best Micro-average AUC:** 0.9717

These results indicate strong classification performance across all three classes.

### Results
![ROC-AUC Curves](images/lens.png)
![ROC-AUC Curves Testing](images/lens.png)

---

## Task VI.A: Masked Autoencoder (MAE) for Feature Representation Learning

### Approach
I implemented a **Masked Autoencoder (MAE)** to learn feature representations from **no_sub samples** of strong lensing images. Each **224×224 image** was divided into **196 patches (16×16 each)**, with a portion of them masked.

### Model Architecture
- **Encoder:** 24-block Transformer processing only visible patches
- **Decoder:** 8-block Transformer reconstructing masked regions
- **Patch Embedding:** Conv2D-based mapping to a **1024-dimensional space**
- **Loss Function:** Mean Squared Error (MSE) over masked patches

### Training Details
- **Initial Loss:** 1.7367
- **Final Training Loss (Epoch 10):** 0.0818
- **Final Validation Loss:** 0.0551

These results suggest the model effectively reconstructs missing parts of input images, demonstrating its potential for pretraining in downstream tasks like lens classification and anomaly detection.

### Results
![MAE Reconstruction](images/lens.png)

---

## Task VI.A: Fine-Tuning MAE for Classification

I fine-tuned the pretrained **MAE encoder** for **multiclass classification** of strong gravitational lensing images by adding a **linear classification head** on top of the **24-block Transformer encoder**.

### Training Strategy
- **Frozen Layers:** Patch embedding layer
- **Trainable Layers:** Transformer encoder
- **Classification Method:** [CLS] token processed through LayerNorm → Linear Head
- **Epochs:** 5

### Performance Metrics
- **Training Accuracy:** 66.83%
- **Validation Accuracy:** 66.60%
- **AUC-ROC Scores:**
  - **Training Set:** 1.0000, 0.7512, 0.7499
  - **Validation Set:** 1.0000, 0.7494, 0.7500

These results indicate successful feature transfer from unsupervised MAE pretraining.

---

## Task VI.B: Fine-Tuning MAE for Super-Resolution

### Approach
I fine-tuned a pretrained **MAE encoder** for **super-resolution**, upscaling images from **(75, 75) to (150, 150)** using a **Transformer-based decoder** and **progressive upsampling**.

### Model Architecture
- **Encoder:** 24-block Transformer extracting low-resolution features
- **Decoder:** 8-block Transformer for feature mapping
- **Upsampling:** PixelShuffle layers + Bilinear interpolation

### Training Details
- **Epochs:** 10
- **Performance Metrics:**
  - **Validation PSNR:** 40.209
  - **Validation SSIM:** 0.9647
  - **Validation MSE:** 0.000096
  - **Training Loss:** Improved from **0.0168 → 0.0111**
  - **Validation Loss:** Reduced from **0.0128 → 0.0111**

These results indicate strong reconstruction quality and effective fine-tuning.

### Results
![Super-Resolution Output]([images/lens.png](https://github.com/shobhit9sept/GSOC---ML4SCI---DeepLense-Task/blob/main/training_curves_SR.png))

---

## Conclusion
This assignment demonstrates my ability to apply **foundation models** to **gravitational lensing tasks**, showcasing:
1. **Strong classification performance** using **ViT**.
2. **Unsupervised representation learning** with **MAE**.
3. **Effective feature transfer** for **downstream classification**.
4. **Super-resolution enhancement** via **Transformer-based upscaling**.

I look forward to further contributing to the **DeepLense project** as part of GSOC 2025.
