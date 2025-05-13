# Strategic Attention-Wise Blocks Integration in Imbalanced Multi-Label Chest X-Ray Diagnosis

This repository contains the official implementation of the paper:

> **Strategic Attention-Wise Blocks Integration in Imbalanced Multi-Label Chest X-Ray Diagnosis**

We propose a two-stage deep learning framework combining VGG and attention-wise blocks to tackle the challenge of imbalanced multi-label classification in chest X-ray diagnosis.

---
![Model Architecture](https://github.com/user-attachments/assets/06c9c433-3365-4f2d-9b86-7da3f3e56a44)

![Training Strategy](https://github.com/user-attachments/assets/8d5b68f4-0bb7-441e-b15b-d12f88ccecd7)

![GradCam](https://github.com/user-attachments/assets/1a7a29d1-56b6-4b11-833b-4dff8265a73b)

![roc_auc_8818](https://github.com/user-attachments/assets/fd7b733f-6c6a-4bac-81bf-b65fbe893623)

---

## 📁 Project Structure

├── train-stage-1-bceloss.ipynb # Stage 1 training notebook using BCE loss

├── train-stage-2-focalloss.ipynb # Stage 2 fine-tuning notebook using Focal loss

├── test-notebook.ipynb # Evaluation notebook (ROC-AUC, predictions)

├── image.png # Model architecture or training pipeline illustration

└── README.md # This file

---

## ⚙️ Setup

- Platform: **Kaggle Notebooks**
- GPU: **P100** (enable in Notebook settings)
- Dependencies: Available in Kaggle by default (fastai, PyTorch, scikit-learn)

---

## 🚀 Training and Evaluation Instructions

### 🔧 1. Train Stage 1 on Kaggle (Backbone Pre-training)

Open `train-stage-1-bceloss.ipynb`:

```python
# Train the model with BCE Loss
lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
learn.fine_tune(freeze_epochs=3,epochs=20, base_lr=lrs.valley)
```
✅ Make sure GPU is set to P100 in Kaggle settings for best performance.

### 🎯 2. Train Stage 2 (Fine-tuning with Focal Loss)
Open `train-stage-2-focalloss.ipynb`:

```python
# Load Stage 1 weights
learn = learn.load('')

# Fine-tune with Focal Loss
learn.unfreeze()
learn.fit_one_cycle(10, slice(2e-5, 8e-5))
```

### 🧪 3. Test and Evaluate the Model
Open `test-notebook.ipynb`:

```python
# Load Stage 2 weights
result = get_roc_auc(model_vgg_lka, 'stage2_model')

# Output ROC-AUC score
print("ROC-AUC Score:", result)
```
---
## 📌 Key Highlights
Stage 1 uses Binary Cross-Entropy Loss to learn abnormal's patterns.

Stage 2 uses Focal Loss to focus on hard and minority classes.

Combines VGG16 backbone with Large Kernel Attention (LKA) modules for better spatial representation.

Designed to handle class imbalance in multi-label medical imaging.

---
## 📜 License
This project is licensed under the MIT License.
Feel free to use, modify, and share with attribution.

---
## 📬 Contact
For questions, feedback, or collaborations, please open an issue on this repository.
