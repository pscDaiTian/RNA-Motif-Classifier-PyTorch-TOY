# RNA Motif Classifier: A PyTorch Toy Model for Sequence Feature Extraction

## 📌 Project Overview
This repository contains an end-to-end PyTorch toy model designed to identify specific regulatory motifs (e.g., "UAUAUA") within RNA sequences. It serves as a proof-of-concept for extracting biological sequence features using deep learning architectures. 

As a researcher with a background in plant epigenetics and multi-omics data analysis (primarily utilizing R), this project marks my active transition into building and debugging Deep Learning pipelines in Python to bridge traditional bioinformatics with AI for Science.

## 🧬 Biological Problem & Dataset
- **Objective:** Classify 50-nt RNA sequences into positive (contains the target motif) and negative (random sequence) categories.
- **Data:** 2,000 synthetic RNA sequences (1,000 positive, 1,000 negative) generated via Python.
- **Preprocessing:** Sequences were converted into numerical matrices using One-Hot Encoding (A=[1,0,0,0], U=[0,1,0,0], etc.).

## 🛠️ Model Evolution & Debugging Journey

This project went through three distinct phases, reflecting the critical differences between general neural networks and biological sequence models:

### V1: Multi-Layer Perceptron (MLP) - The "Overfitting" Trap
- **Architecture:** `nn.Linear` layers with ReLU activation.
- **Result:** Training loss dropped to near zero, but Test Accuracy plateaued around 60-65%.
- **Biological Insight:** The MLP lacks translational invariance. It memorized the absolute position of the motif in the training set and failed when the motif "slid" to a different position in the testing set.

### V2: 1D-CNN (Initial Attempt) - The Tensor Reshaping Bug
- **Architecture:** Introduced `nn.Conv1d` (acting as a sliding PWM scanner) and `AdaptiveMaxPool1d`.
- **Result:** Accuracy remained low.
- **Debugging:** Discovered a critical tensor reshaping error. Using `.view(-1, 4, 50)` directly on the flattened 1D array essentially "shredded" the sequence logic, distributing consecutive nucleotides across different channels, destroying the biological continuity of the motif.

### V3: 1D-CNN (Final Corrected Version) - Success 🚀
- **Fix:** Correctly restored the biological structure `(Batch, 50, 4)` first, then applied `.transpose(1, 2)` to meet PyTorch's `(Batch, Channels, Length)` requirement.
- **Result:** The 1D-CNN successfully scanned the sequence regardless of motif position.
- **Performance:** **Test Accuracy reached > 99%** within 5 epochs.

## 🚀 How to Run
1. Clone this repository.
2. Ensure you have `torch`, `pandas`, and `numpy` installed.
3. Run the `RNA_Motif_Classifier_PyTorch.ipynb` notebook cell by cell.

## 💡 Future Directions
This toy model demonstrates the fundamental logic of sequence embedding and convolutional feature extraction. My next steps involve exploring Transformer-based architectures (self-attention mechanisms) and applying them to real-world, high-throughput transcriptomic datasets.
