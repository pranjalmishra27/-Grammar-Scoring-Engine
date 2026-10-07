# 🎙️ Grammar Scoring Engine for Spoken Audio

A multi-modal machine learning pipeline designed for the **SHL Hiring Assessment Challenge** to predict continuous grammar and fluency proficiency scores ($0.0 - 5.0$) from spoken audio clips ($45 - 60$ seconds).

---

## 🌟 Overview & Architecture

Grammar and spoken fluency live at the intersection of **acoustic flow** and **linguistic syntax**. A pure acoustic or pure text model is sub-optimal; this repository combines a multi-view hierarchical architecture:

```
                  ┌─► Whisper ASR ────────► Handcrafted Syntactic & Fluency Features ──┐
                  │                                                                     │
Audio Clip (.wav) ┼─► CoLA Grammar Model ──► Grammatical Acceptability Probabilities ──┼─► 10-Fold Multi-Model Ensemble ─► Dual Meta-Stacker ─► Score
(16 kHz Mono)     │                                                                     │   (LightGBM, Ridge, SVR, HGB)      (NNLS + Ridge)    [1.0, 5.0]
                  ├─► Librosa Prosody ────► 20 MFCCs, RMS dynamics, ZCR, Spectrals ────┤
                  │                                                                     │
                  └─► Wav2Vec2 Embeddings ─► Multi-Scale Acoustic Hidden States ────────┘
```

---

## 📊 Pipeline Evolution & Leaderboard Progress

| Version / Script | Key Innovation | CV RMSE | Public LB RMSE |
| :--- | :--- | :---: | :---: |
| **`pipeline.py` (V1)** | Baseline Whisper + Wav2Vec2 + 5-Fold Ridge/SVR Stacker | `0.6205` | `0.4783` |
| **`rmse_pipeline.py` (V2)** | 10-Fold CV + Acoustic Prosody + Strict NNLS L2 Stacking | `0.5945` | `0.4604` (PB) |
| **`ultra_pipeline.py` (V3)** | Multi-Seed 10-Fold CV (3 Seeds) + Interaction Features + Dual Meta Blend | `0.5897` | `0.4572` (PB) |
| **`blend_pipeline.py` (V4)** | CoLA Grammar Probabilities + Multi-Seed 40-Fold Master Blend | `0.5897` | Generated `submission3.csv` |

---

## 🚀 Key Technical Features

### 1. Acoustic & Prosodic Feature Extraction
* **20 MFCCs across time:** Mean, Standard Deviation, and Skewness (capturing spectral timbre and vocal resonance).
* **Energy Dynamics:** RMS Mean, RMS Variance, Peak Dynamic Range.
* **Articulation & Phonation:** Zero-Crossing Rate (ZCR), Spectral Centroid, Spectral Rolloff, and Spectral Flatness.

### 2. Linguistic, Syntactic & Grammar Signal
* **CoLA Grammar Acceptability:** Zero-shot syntax probability scoring using `textattack/roberta-base-CoLA`.
* **Lexical Diversity:** Type-Token Ratio (TTR), Root-TTR, Bilogarithmic TTR ($\frac{\log V}{\log N}$).
* **Syntactic Complexity:** Subordinating/coordinating conjunction density, transitional adverbs, sentence length variance, and pause gaps.
* **Fluency Dynamics:** Immediate word repetitions (stutters), multi-word fillers (`"you know"`, `"like"`, `"basically"`), and agreement error heuristics.

### 3. Representation Learning
* **Text Semantics:** `sentence-transformers/all-MiniLM-L6-v2` dense embeddings.
* **Acoustic Speech:** `facebook/wav2vec2-base-960h` mean & std pooled representations with multi-scale PCA decompositions (48, 96, 160 components).

### 4. Ensembling & Meta-Stacking
* **10-Fold Stratified Cross-Validation across 4 random seeds** (40 total models per architecture).
* **Candidate Pool:** LightGBM, HistGradientBoosting, RandomForest, ExtraTrees, SVR, RidgeCV, and HuberRegressor.
* **Non-Negative Least Squares (NNLS) Meta-Learner:** Mathematically eliminates collinearity and minimizes quadratic residual error.
* **Optimal Scale & Shrinkage Calibration:** Corrects conditional variance to eliminate out-of-distribution penalties.

---

## 📁 Repository Structure

```
├── grammar_scoring_engine.ipynb   # Interactive Jupyter Notebook for exploration & visuals
├── pipeline.py                    # V1 baseline end-to-end pipeline
├── rmse_pipeline.py               # V2 strict RMSE-minimizing pipeline
├── ultra_pipeline.py              # V3 multi-seed 10-fold CV pipeline
├── blend_pipeline.py              # V4 CoLA grammar-augmented master blend
├── submission.csv                 # V2 submission predictions (0.4604)
├── submission2.csv                # V3 submission predictions (0.4572)
├── submission3.csv                # V4 master blend submission predictions
└── README.md                      # Project documentation
```

---

## 💻 Quick Start

### 1. Requirements
```bash
pip install torch torchaudio librosa faster-whisper transformers sentence-transformers lightgbm scikit-learn pandas numpy tqdm scipy
```

### 2. Run the Full Pipeline
```bash
python blend_pipeline.py
```
Outputs final predictions to `submission3.csv`.

---

## 🏆 Author
* **Pranjal Mishra**
