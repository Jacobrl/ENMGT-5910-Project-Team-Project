# ENMGT-5910 Car Damage Detection Pipeline

## 📁 Project Overview
This repository hosts our group project for building an end-to-end AI pipeline that:
1. Recognizes car **make, model, and year**
2. Detects **vehicle damages**
3. Estimates **repair cost** (in USD and pesos)
4. Generates **structured claim descriptions**

Each stage is modular, located under `src/`:
- `src/recognition/` → Car recognition (ResNet50 baseline; ViT later)
- `src/detection/` → Damage detection (YOLO/Mask R-CNN)
- `src/cost_model/` → Cost estimation (XGBoost/MLP/Hybrid)
- `src/description/` → Claim description generation (BLIP-2/LLaVA/API)

---

## ⚙️ Environment Setup
To replicate the same environment locally:
```bash
# Clone the repository
git clone https://github.com/Jacobrl/ENMGT-5910-Project-Team-Project.git
cd ENMGT-5910-Project-Team-Project

# Create and activate the environment
conda env create -f environment.yml
conda activate carpipeline