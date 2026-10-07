import os
import re
import sys
import json
import time
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import RidgeCV, LinearRegression, ElasticNetCV, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
SEEDS = [42, 123, 777, 2026]
DATA_DIR = r"e:\SHL\data"
WORK_DIR = r"e:\SHL\work"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*75)
print("     BLEND & GRAMMAR-AUGMENTED RMSE OPTIMIZATION PIPELINE (V4)")
print("     Target Output: submission3.csv")
print("="*75)

# 1. Load Data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
FILE_COL = "filename"
LABEL_COL = "label"

y = train_df[LABEL_COL].values.astype(float)
def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

baseline_rmse = np.sqrt(np.mean((y - y.mean())**2))
print(f"Train: {len(train_df)} | Test: {len(test_df)}")
print(f"Baseline Mean-Predictor RMSE: {baseline_rmse:.4f}\n")

# Load cached transcripts
TR_CACHE = os.path.join(WORK_DIR, "transcripts.json")
transcripts = json.load(open(TR_CACHE, "r", encoding="utf-8"))
train_df["text"] = [transcripts[f"train_{fn}"]["text"] for fn in train_df[FILE_COL]]
test_df["text"]  = [transcripts[f"test_{fn}"]["text"]  for fn in test_df[FILE_COL]]

# ==========================================
# 2. EXTRACT CoLA GRAMMAR ACCEPTABILITY SCORES
# ==========================================
COLA_CACHE = os.path.join(WORK_DIR, "cola_features.npz")

if os.path.exists(COLA_CACHE):
    print("Loading cached CoLA grammar acceptability features...")
    c_data = np.load(COLA_CACHE)
    X_cola_tr = c_data["X_cola_tr"]
    X_cola_te = c_data["X_cola_te"]
else:
    print("Extracting CoLA grammatical acceptability probabilities from transcripts...")
    model_name = "textattack/roberta-base-CoLA"
    cola_tok = AutoTokenizer.from_pretrained(model_name)
    cola_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE).eval()
    
    @torch.no_grad()
    def extract_cola_score(text):
        sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if not sents:
            return [0.5, 0.5, 0.0, 0.0]
        
        inputs = cola_tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
        logits = cola_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy() # prob of being grammatically acceptable
        
        mean_p = float(np.mean(probs))
        min_p  = float(np.min(probs))
        bad_frac = float(np.mean(probs < 0.5))
        mean_logit = float(np.mean(logits[:, 1].cpu().numpy()))
        return [mean_p, min_p, bad_frac, mean_logit]

    cola_tr = [extract_cola_score(t) for t in tqdm(train_df["text"], desc="Train CoLA")]
    cola_te = [extract_cola_score(t) for t in tqdm(test_df["text"],  desc="Test CoLA")]
    
    X_cola_tr = np.array(cola_tr)
    X_cola_te = np.array(cola_te)
    np.savez_compressed(COLA_CACHE, X_cola_tr=X_cola_tr, X_cola_te=X_cola_te)
    del cola_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print(f"CoLA Grammar features: {X_cola_tr.shape}")

# Load previous features
feat_cache = np.load(os.path.join(WORK_DIR, "all_features.npz"))
X_hand_tr = feat_cache["X_hand_tr"]
X_hand_te = feat_cache["X_hand_te"]
X_txt_tr  = feat_cache["X_txt_tr"]
X_txt_te  = feat_cache["X_txt_te"]
X_aud_tr  = feat_cache["X_aud_tr"]
X_aud_te  = feat_cache["X_aud_te"]

p_data = np.load(os.path.join(WORK_DIR, "prosody_features.npz"))
X_pro_tr = p_data["X_pro_tr"]
X_pro_te = p_data["X_pro_te"]

# Interaction Features
def make_interactions(tab_mat, pro_mat, cola_mat):
    dur = pro_mat[:, 0]
    rms_dyn = pro_mat[:, 4]
    zcr_mean = pro_mat[:, 5]
    
    extra = np.column_stack([
        tab_mat[:, 2] * tab_mat[:, 4],       # wps * ttr
        rms_dyn / (dur + 1e-4),              # dynamic energy rate
        zcr_mean * tab_mat[:, 2],            # articulation dynamics
        tab_mat[:, 11] * pro_mat[:, 1],      # filler rate * rms mean
        cola_mat[:, 0] * tab_mat[:, 4],      # grammar prob * ttr
        cola_mat[:, 1] * (1.0 - tab_mat[:, 11]), # min grammar * (1 - filler rate)
    ])
    return np.hstack([tab_mat, pro_mat, cola_mat, extra])

X_tab_full_tr = make_interactions(X_hand_tr, X_pro_tr, X_cola_tr)
X_tab_full_te = make_interactions(X_hand_te, X_pro_te, X_cola_te)
print(f"Full Augmented Tabular Matrix: train={X_tab_full_tr.shape}, test={X_tab_full_te.shape}")

# Scaled Audio representations
scaler_aud = StandardScaler()
X_aud_scaled_tr = scaler_aud.fit_transform(X_aud_tr)
X_aud_scaled_te = scaler_aud.transform(X_aud_te)

pca_48 = PCA(n_components=48, random_state=42)
X_aud_pca48_tr = pca_48.fit_transform(X_aud_scaled_tr)
X_aud_pca48_te = pca_48.transform(X_aud_scaled_te)

pca_96 = PCA(n_components=96, random_state=42)
X_aud_pca96_tr = pca_96.fit_transform(X_aud_scaled_tr)
X_aud_pca96_te = pca_96.transform(X_aud_scaled_te)

pca_160 = PCA(n_components=160, random_state=42)
X_aud_pca160_tr = pca_160.fit_transform(X_aud_scaled_tr)
X_aud_pca160_te = pca_160.transform(X_aud_scaled_te)

# Concatenated full multi-view
scaler_tab = StandardScaler()
X_all_raw_tr = np.hstack([scaler_tab.fit_transform(X_tab_full_tr), X_txt_tr, X_aud_pca96_tr])
X_all_raw_te = np.hstack([scaler_tab.transform(X_tab_full_te), X_txt_te, X_aud_pca96_te])

pca_all = PCA(n_components=120, random_state=42)
X_all_pca_tr = pca_all.fit_transform(X_all_raw_tr)
X_all_pca_te = pca_all.transform(X_all_raw_te)

# ==========================================
# 3. MULTI-SEED 10-FOLD MODEL SUITE
# ==========================================
alphas_dense = np.logspace(-3, 4, 40)

def get_models(seed):
    return {
        "tab_lgb_a": (
            LGBMRegressor(n_estimators=220, learning_rate=0.025, num_leaves=14, max_depth=4, subsample=0.8, colsample_bytree=0.75, reg_alpha=1.5, reg_lambda=3.0, random_state=seed, verbose=-1),
            X_tab_full_tr, X_tab_full_te
        ),
        "tab_lgb_b": (
            LGBMRegressor(n_estimators=160, learning_rate=0.035, num_leaves=10, max_depth=3, subsample=0.85, colsample_bytree=0.85, reg_alpha=2.0, reg_lambda=4.0, random_state=seed+1, verbose=-1),
            X_tab_full_tr, X_tab_full_te
        ),
        "tab_hgb": (
            HistGradientBoostingRegressor(max_depth=3, learning_rate=0.025, max_iter=220, l2_regularization=3.0, random_state=seed),
            X_tab_full_tr, X_tab_full_te
        ),
        "tab_ridge": (
            make_pipeline(StandardScaler(), RidgeCV(alphas=alphas_dense)),
            X_tab_full_tr, X_tab_full_te
        ),
        "tab_et": (
            make_pipeline(StandardScaler(), ExtraTreesRegressor(n_estimators=160, max_depth=7, min_samples_split=4, random_state=seed, n_jobs=-1)),
            X_tab_full_tr, X_tab_full_te
        ),
        "txt_ridge": (
            make_pipeline(StandardScaler(), RidgeCV(alphas=alphas_dense*10)),
            X_txt_tr, X_txt_te
        ),
        "txt_svr": (
            make_pipeline(StandardScaler(), SVR(C=1.2, epsilon=0.08, gamma="scale")),
            X_txt_tr, X_txt_te
        ),
        "aud_ridge_96": (
            make_pipeline(StandardScaler(), RidgeCV(alphas=alphas_dense)),
            X_aud_pca96_tr, X_aud_pca96_te
        ),
        "aud_ridge_160": (
            make_pipeline(StandardScaler(), RidgeCV(alphas=alphas_dense)),
            X_aud_pca160_tr, X_aud_pca160_te
        ),
        "aud_svr_48": (
            make_pipeline(StandardScaler(), SVR(C=1.2, epsilon=0.08, gamma="scale")),
            X_aud_pca48_tr, X_aud_pca48_te
        ),
        "aud_lgb": (
            LGBMRegressor(n_estimators=160, learning_rate=0.025, num_leaves=10, max_depth=3, subsample=0.8, colsample_bytree=0.7, reg_alpha=2.0, reg_lambda=4.0, random_state=seed, verbose=-1),
            X_aud_pca48_tr, X_aud_pca48_te
        ),
        "all_ridge": (
            make_pipeline(StandardScaler(), RidgeCV(alphas=alphas_dense)),
            X_all_pca_tr, X_all_pca_te
        ),
        "all_huber": (
            make_pipeline(StandardScaler(), HuberRegressor(max_iter=400, alpha=30.0)),
            X_all_pca_tr, X_all_pca_te
        ),
    }

print("\n--- Running Multi-Seed 10-Fold Cross-Validation (4 Seeds, 40 Folds Total) ---")
N_FOLDS = 10
strat_bins = np.clip(np.round(y), 1, 5).astype(int)

seed_oof_preds = {k: np.zeros(len(y)) for k in get_models(42).keys()}
seed_test_preds = {k: np.zeros(len(test_df)) for k in get_models(42).keys()}

for seed in SEEDS:
    print(f">> Processing Seed {seed}...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(y)), strat_bins))
    models_dict = get_models(seed)
    
    for name, (model, Xtr, Xte) in models_dict.items():
        p_oof = np.zeros(len(y))
        p_test = np.zeros(len(test_df))
        
        for tr_idx, val_idx in folds:
            m = model.fit(Xtr[tr_idx], y[tr_idx])
            p_oof[val_idx] = m.predict(Xtr[val_idx])
            p_test += m.predict(Xte) / N_FOLDS
            
        seed_oof_preds[name] += p_oof / len(SEEDS)
        seed_test_preds[name] += p_test / len(SEEDS)

print("\n" + "="*75)
print("MULTI-SEED OOF EVALUATION SUMMARY")
print("="*75)
for name in seed_oof_preds:
    m_rmse = rmse(y, seed_oof_preds[name])
    m_pear = pearsonr(y, seed_oof_preds[name])[0]
    print(f"Model: {name:15s} | 4-Seed OOF RMSE: {m_rmse:.4f} | Pearson r: {m_pear:.4f}")

# ==========================================
# 4. HIERARCHICAL STACKING & SUBMISSION ENSEMBLING
# ==========================================
print("\n" + "="*75)
print("4. HIERARCHICAL META-STACKING & MULTI-PIPELINE BLEND")
print("="*75)

OOF_MAT = np.column_stack([seed_oof_preds[k] for k in seed_oof_preds])
TEST_MAT = np.column_stack([seed_test_preds[k] for k in seed_test_preds])

# NNLS Meta-Model
meta_nnls_full = LinearRegression(positive=True).fit(OOF_MAT, y)
current_test_pred = meta_nnls_full.predict(TEST_MAT)

# Cross-Validated NNLS OOF
nnls_cv_oof = np.zeros(len(y))
skf_meta = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for tr_idx, val_idx in skf_meta.split(OOF_MAT, strat_bins):
    m = LinearRegression(positive=True).fit(OOF_MAT[tr_idx], y[tr_idx])
    nnls_cv_oof[val_idx] = m.predict(OOF_MAT[val_idx])

print(f"Augmented NNLS Meta CV RMSE: {rmse(y, nnls_cv_oof):.4f} (Pearson: {pearsonr(y, nnls_cv_oof)[0]:.4f})")

# Load previous successful submissions
sub1_path = os.path.join(r"e:\SHL", "submission.csv")
sub2_path = os.path.join(r"e:\SHL", "submission2.csv")

pred1 = pd.read_csv(sub1_path)["label"].values if os.path.exists(sub1_path) else current_test_pred
pred2 = pd.read_csv(sub2_path)["label"].values if os.path.exists(sub2_path) else current_test_pred

# Weighted Master Blend: 50% New CoLA-Augmented V4 + 35% V3 (submission2) + 15% V2 (submission)
final_master_blend = 0.50 * current_test_pred + 0.35 * pred2 + 0.15 * pred1

# Calibration & Optimal Shrinkage for RMSE
calib_a = 1.045
calib_b = -0.150
calibrated_final = np.clip(calib_a * final_master_blend + calib_b, 1.40, 5.00)

# Exact mean match to train target
shift = np.mean(y) - np.mean(calibrated_final)
calibrated_final = np.clip(calibrated_final + shift, 1.40, 5.00)

print(f"\nFinal Blend Statistics:")
print(f"  Mean : {calibrated_final.mean():.4f} (Train: {y.mean():.4f})")
print(f"  Std  : {calibrated_final.std():.4f}")
print(f"  Range: [{calibrated_final.min():.3f}, {calibrated_final.max():.3f}]")

# ==========================================
# 5. SAVE TO submission3.csv
# ==========================================
sub3_df = pd.DataFrame({
    "filename": test_df[FILE_COL],
    "label": calibrated_final
})

sub3_path = r"e:\SHL\submission3.csv"
sub3_df.to_csv(sub3_path, index=False)

print("\n" + "="*75)
print(f"SUCCESSFULLY GENERATED AND SAVED: {sub3_path}")
print(f"Total Rows: {len(sub3_df)}")
print("="*75)
print("\nFirst 10 rows of submission3.csv:")
print(sub3_df.head(10))
