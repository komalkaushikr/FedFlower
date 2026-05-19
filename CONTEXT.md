# FedFlower — Full Project Context for Claude Code

## Stack
ResNet50 + Transfer Learning, Oxford 102 Flowers, PyTorch, FedAvg, Google Colab, Android (PyTorch Mobile)

## Confirmed results
- best_model.pth = 89.04% Top-1 accuracy on 6,149 test images
- Dataset: 1,020 train + 1,020 val = 2,040 for federated split. 6,149 test = never touched.
- Phase 2 ran and showed 93.5% federated accuracy (because it fine-tuned ON TOP of best_model.pth, not from scratch — this is intentional and documented)

## Architecture
- Model: ResNet50 pretrained on ImageNet, final FC layer replaced for 102 classes
- Aggregation: FedAvg — global = weighted average of client weights by data size
- Local training: 3 epochs per client per round, 10 rounds total
- Input normalization: mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
- BatchNorm MUST be frozen before export (model.eval() + explicit BN freeze)

## What Phase 2 must do (rewrite to include ALL of this)
1. Upload best_model.pth at start
2. Strong augmentation on training data:
   RandomHorizontalFlip, RandomVerticalFlip, ColorJitter(0.3,0.3,0.3),
   RandomRotation(30), RandomResizedCrop(224, scale=(0.7,1.0))
   Val/test: only Resize(256) + CenterCrop(224) + normalize
3. Run 5-client simulation (408 imgs each), 10 rounds
4. Run 10-client simulation (204 imgs each), 10 rounds
5. Per round: print and save cross-entropy loss per client + global average loss
6. After training: print per-client accuracy on (a) client's own local data, (b) global 6149 test set
7. Print F1 score (macro, sklearn) per client on global test set
8. Plot 1: Federated accuracy vs rounds (5-client line + 10-client line + centralized 89.04% dashed)
9. Plot 2: Cross-entropy loss per round (5-client avg + 10-client avg)
10. Download: federated_results.json, federated_accuracy.png, federated_loss.png

## What Phase 3 needs (keep mostly same, verify these exist)
1. Upload best_model.pth
2. Top-1, Top-5 accuracy on 6,149 test images
3. Macro F1-score
4. Confusion matrix (102x102, saved as PNG)
5. Per-class accuracy bar chart
6. Grad-CAM on 5 sample images (heatmap overlay saved as PNG)
7. Download all PNGs

## What Phase 4 must do
1. Upload best_model.pth
2. Freeze ALL BatchNorm layers explicitly:
   for m in model.modules():
       if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)): m.eval()
3. model.eval()
4. torch.jit.trace with dummy input shape [1,3,224,224]
5. Save as flower_traced.pt
6. Download flower_traced.pt

## Android app changes needed
1. Load flower_traced.pt from assets
2. Confidence threshold = 35%
3. If top-1 confidence < 35%: show "Flower not recognized. Try a closer, well-lit photo."
4. If >= 35%: show top 3 predictions with name + confidence %
5. Input preprocessing must match exactly:
   Resize 256 → CenterCrop 224 → float/255 → normalize with ImageNet mean/std

## Files that must be downloaded and kept on your laptop
From Phase 1: best_model.pth, centralized_results.json
From Phase 2: federated_results.json, federated_accuracy.png, federated_loss.png
From Phase 3: confusion_matrix.png, per_class_accuracy.png, gradcam_results.png
From Phase 4: flower_traced.pt (goes into Android assets/)