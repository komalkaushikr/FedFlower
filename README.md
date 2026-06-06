# FedFlower: Federated Fine-Tuning for Decentralized On-Device Flower Species Recognition

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange) ![Android](https://img.shields.io/badge/Android-Java-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end machine learning system that identifies **102 flower species** from photos — trained using **Federated Learning** so data storage is distributed across clients instead of a central server — and deployed as an **Android app** with fully on-device inference. No internet required.

---

## Why Federated Learning?

Traditional ML requires storing the entire training dataset on a single server. As datasets grow into hundreds of gigabytes, this becomes expensive and hard to scale. FedFlower distributes both **data storage and training** across multiple clients — each client holds only its own subset, and the server never stores raw images.

```
Centralized:                      Federated:
  All 2,040 images on 1 node        408 images × 5 clients
  Server stores everything          Server stores only weights
```

---

## Results

| Configuration | Top-1 Accuracy | Macro F1 | Per-client storage |
|---|---|---|---|
| Centralized baseline | 87.92% | 0.8753 | 2,040 images |
| Federated (5 clients, 10 rounds) | **91.82%** | **0.9159** | 408 images |
| Federated (10 clients, 10 rounds) | 90.65% | 0.9050 | 204 images |

> The federated model was **warm-started from the centralized checkpoint** and refined through 10 federated rounds, raising accuracy by +3.90 percentage points while reducing per-client storage to roughly one-fifth of the full pool.

---

## App Performance

- 📷 Camera or gallery input
- ⚡ ~340 ms inference on Motorola Edge 50
- 🌸 Top-3 predictions with confidence scores
- 🚫 "Flower not recognized" fallback below 35% confidence
- 🔒 Fully offline — no data sent anywhere

---

## Model Architecture

ResNet50 pretrained on ImageNet, with layers 1–3 frozen (generic features preserved) and layer 4 + custom head fine-tuned for flowers:

```
Linear(2048 → 512) → BatchNorm → ReLU → Dropout(0.4) → Linear(512 → 102)
```

---

## How FedAvg Works

Each communication round:
1. Server sends the current global model to every client
2. Each client trains it locally on **its own data** for 3 epochs
3. Each client sends back only the **updated weights** (never raw images)
4. Server computes the weighted average: `W_global = Σ (nᵢ/N) × Wᵢ`
5. Repeat for 10 rounds

---

## Project Structure

```
FedFlower/
├── notebooks/
│   ├── Phase1_Centralized_CNN.ipynb       ← ResNet50 baseline → 87.92%
│   ├── Phase2_Federated_Learning.ipynb    ← FedAvg, 5- and 10-client
│   ├── Phase3_Evaluation_GradCAM.ipynb    ← Confusion matrix, Grad-CAM
│   └── Phase4_Model_Conversion.ipynb      ← Export to PyTorch Mobile
├── android/
│   └── app/src/main/
│       ├── java/com/fedflower/app/MainActivity.java
│       ├── assets/flower_traced.pt
│       └── assets/flower_names.txt
├── results/
└── README.md
```

---

## How to Run

### Training (Google Colab)
1. Open notebooks from this repo in Colab
2. Runtime → T4 GPU
3. Run Phase 1 → 2 → 3 → 4
4. Download `flower_traced.pt` after Phase 4

### Android App
1. Open `android/` in Android Studio
2. Connect your phone (USB debugging on)
3. Click **Run**

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch 2.1, torchvision |
| Backbone | ResNet50 (ImageNet pretrained) |
| Dataset | Oxford 102 Flowers (8,189 images, 102 species) |
| Federated Algorithm | FedAvg (Federated Averaging) |
| Mobile Deployment | PyTorch Mobile |
| Android App | Java, Android Studio |

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*Built with PyTorch · Federated Learning · Android · Oxford 102 Flowers*
