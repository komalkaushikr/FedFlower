ASSETS FOLDER — FedFlower Android App
=======================================

This folder must contain TWO files before building the app:

1. flower_traced.pt  ← DOWNLOAD FROM COLAB AFTER RUNNING Phase4_Model_Conversion.ipynb
   - TorchScript traced model for PyTorch Mobile (torch.jit.trace)
   - Size: ~100 MB
   - How to get it: Run Phase4_Model_Conversion.ipynb in Colab, then download
     flower_traced.pt when prompted in the final cell

2. flower_names.txt  ← ALREADY INCLUDED
   - 102 flower species names, one per line
   - Matches Oxford 102 Flowers label ordering (0-indexed)

WITHOUT flower_traced.pt, the app will crash on launch with a FileNotFoundException.
