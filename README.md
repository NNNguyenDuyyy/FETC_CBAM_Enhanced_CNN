# Improving Imbalanced Multi-Label Chest X-Ray Diagnosis via CBAM-Enhanced CNN Backbones

This repository contains the official implementation of the paper:

> **Improving Imbalanced Multi-Label Chest X-Ray Diagnosis via CBAM-Enhanced CNN Backbones**

We propose a deep learning framework combining CNN and attention-wise blocks to tackle the challenge of imbalanced multi-label classification in chest X-ray diagnosis.

---
![Overall Architecture](https://github.com/user-attachments/assets/6d8e426b-afb5-45e3-91fa-bd7cdd446312)
![densenet](https://github.com/user-attachments/assets/41d11018-46d9-4937-99a5-7fbf10c67b50)
![vgg](https://github.com/user-attachments/assets/7783115f-f5ad-4c44-bfb1-0dc0402186a9)
<img width="3572" height="3184" alt="dense_larger" src="https://github.com/user-attachments/assets/a6767300-15d9-42e6-b2b6-11148b51d981" />
<img width="3572" height="3184" alt="vgg_larger" src="https://github.com/user-attachments/assets/9ca1a326-0a2f-4807-a5f9-4c0a05df1872" />

---

## 📁 Project Structure

├── FETC-2025-CBAM-Enhanced-CNN-Train-Stage-1.ipynb # Stage 1: training notebook using BCE loss

├── FETC-2025-CBAM-Enhanced-CNN-Train-Stage-2.ipynb # Stage 2: fine-tuning notebook 

├── FETC-2025-CBAM-Enhanced-CNN-Testing.ipynb # Evaluation notebook 

├── FETC-2025-CBAM-Enhanced-CNN-Weights # Trained Weights for Testing

├── LICENSE 

└── README.md 

---

## ⚙️ Setup

- Platform: **Kaggle Notebooks**
- GPU: **P100** (enable in Notebook settings)
- Dependencies: Available in Kaggle by default (fastai, PyTorch, scikit-learn)

---

## 🚀 Training and Evaluation Instructions

### 🔧 1. Train Stage 1 on Kaggle (Backbone Pre-training)

Open `FETC-2025-CBAM-Enhanced-CNN-Train-Stage-1.ipynb`:

```python
# Train the model with BCE Loss
lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
learn.fine_tune(freeze_epochs=3,epochs=20, base_lr=lrs.valley)
```
✅ Make sure GPU is set to P100 in Kaggle settings for best performance.

### 🎯 2. Train Stage 2 (Fine-tuning with Focal Loss)
Open `FETC-2025-CBAM-Enhanced-CNN-Train-Stage-2.ipynb`:

```python
# Load Stage 1 weights
learn = learn.load('')

# Fine-tune with Focal Loss
learn.unfreeze()
learn.fit_one_cycle(10, slice(2e-5, 8e-5))
```

### 🧪 3. Test and Evaluate the Model
Open `FETC-2025-CBAM-Enhanced-CNN-Testing.ipynb`:

```python
# Load Stage 2 weights
result = get_roc_auc(..., '...')  # remove the ".pth"

# Output ROC-AUC score
print("ROC-AUC Score:", result)
```
---
## 📌 Key Highlights
Combine CNN backbone with Convolutional Block Attention Module (CBAM) modules for better spatial representation.
Find the appropriate to inject CBAM for the best results.
Do ablation study for better insights the two stage training strategy.

---
## 📜 License
This project is licensed under the MIT License.
Feel free to use, modify, and share with attribution.

---
## 📬 Contact
For questions, feedback, or collaborations, please open an issue on this repository.
