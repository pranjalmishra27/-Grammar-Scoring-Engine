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
SEEDS = [42, 123, 777]
DATA_DIR = r"e:\SHL\data"
WORK_DIR = r"e:\SHL\work"

print("="*75)
print("     ULTRA MULTI-SEED RMSE OPTIMIZATION PIPELINE (V3)")
print("     Target Output: submission2.csv")
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
print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")
print(f"Baseline Mean-Predictor RMSE: {baseline_rmse:.4f}\n")

# Load cached features
feat_cache = np.load(os.path.join(WORK_DIR, "all_features.npz"))
X_hand_tr = feat_cache["X_hand_tr"]
X_hand_te = feat_cache["X_hand_te"]
X_txt_tr  = feat_cache["X_txt_tr"]
X_txt_te  = feat_cache["X_txt_te"]
X_aud_tr  = feat_cache["X_aud_tr"]
X_aud_te  = feat_cache["X_aud_te"]

# Load prosody features
p_data = np.load(os.path.join(WORK_DIR, "prosody_features.npz"))
X_pro_tr = p_data["X_pro_tr"]
X_pro_te = p_data["X_pro_te"]

# Engineer interaction features
def make_interactions(tab_mat, pro_mat):
    # tab_mat columns: wps, ttr, avg_slen, filler_rate, repeat_rate, etc.
    dur = pro_mat[:, 0]
    rms_dyn = pro_mat[:, 4]
    zcr_mean = pro_mat[:, 5]
    
    extra = np.column_stack([
        tab_mat[:, 2] * tab_mat[:, 4], # wps * ttr
        rms_dyn / (dur + 1e-4),        # dynamic energy rate
        zcr_mean * tab_mat[:, 2],      # articulation dynamics
        tab_mat[:, 11] * pro_mat[:, 1],# filler rate * rms mean
    ])
    return np.hstack([tab_mat, pro_mat, extra])

X_tab_full_tr = make_interactions(X_hand_tr, X_pro_tr)
X_tab_full_te = make_interactions(X_hand_te, X_pro_te)
print(f"Full Tabular + Interaction Features: train={X_tab_full_tr.shape}, test={X_tab_full_te.shape}")

# Scale Audio representations with multi-scale PCA
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

pca_all = PCA(n_components=110, random_state=42)
X_all_pca_tr = pca_all.fit_transform(X_all_raw_tr)
X_all_pca_te = pca_all.transform(X_all_raw_te)

# ==========================================
# 2. MULTI-SEED 10-FOLD CROSS-VALIDATION
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
            make_pipeline(StandardScaler(), ExtraTreesRegressor(n_estimators=150, max_depth=7, min_samples_split=5, random_state=seed, n_jobs=-1)),
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

print("\n--- Running Multi-Seed 10-Fold Cross-Validation ---")
N_FOLDS = 10
strat_bins = np.clip(np.round(y), 1, 5).astype(int)

seed_oof_preds = {k: np.zeros(len(y)) for k in get_models(42).keys()}
seed_test_preds = {k: np.zeros(len(test_df)) for k in get_models(42).keys()}

for seed in SEEDS:
    print(f"\n>> Processing Seed {seed}...")
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
print("MULTI-SEED OOF MODEL EVALUATION SUMMARY")
print("="*75)
for name in seed_oof_preds:
    m_rmse = rmse(y, seed_oof_preds[name])
    m_pear = pearsonr(y, seed_oof_preds[name])[0]
    print(f"Model: {name:15s} | Multi-Seed OOF RMSE: {m_rmse:.4f} | Pearson r: {m_pear:.4f}")

# ==========================================
# 3. DUAL-META ENSEMBLE (NNLS + RIDGE BLEND)
# ==========================================
print("\n" + "="*75)
print("3. DUAL-META ENSEMBLE OPTIMIZATION")
print("="*75)

OOF_MAT = np.column_stack([seed_oof_preds[k] for k in seed_oof_preds])
TEST_MAT = np.column_stack([seed_test_preds[k] for k in seed_test_preds])

# Meta-Model A: Non-Negative Least Squares (NNLS)
nnls_oof = np.zeros(len(y))
skf_meta = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for tr_idx, val_idx in skf_meta.split(OOF_MAT, strat_bins):
    meta_nnls = LinearRegression(positive=True).fit(OOF_MAT[tr_idx], y[tr_idx])
    nnls_oof[val_idx] = meta_nnls.predict(OOF_MAT[val_idx])

meta_nnls_full = LinearRegression(positive=True).fit(OOF_MAT, y)
nnls_test = meta_nnls_full.predict(TEST_MAT)

# Meta-Model B: Regularized Ridge Meta-Regressor
ridge_oof = np.zeros(len(y))
for tr_idx, val_idx in skf_meta.split(OOF_MAT, strat_bins):
    meta_ridge = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(OOF_MAT[tr_idx], y[tr_idx])
    ridge_oof[val_idx] = meta_ridge.predict(OOF_MAT[val_idx])

meta_ridge_full = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(OOF_MAT, y)
ridge_test = meta_ridge_full.predict(TEST_MAT)

# Blended Meta Prediction (65% NNLS + 35% Regularized Ridge)
final_oof_blend = 0.65 * nnls_oof + 0.35 * ridge_oof
final_test_blend = 0.65 * nnls_test + 0.35 * ridge_test

final_oof_blend = np.clip(final_oof_blend, 0.0, 5.0)
final_test_blend = np.clip(final_test_blend, 0.0, 5.0)

final_cv_rmse = rmse(y, final_oof_blend)
final_cv_pear = pearsonr(y, final_oof_blend)[0]

print(f"NNLS Meta CV RMSE    : {rmse(y, nnls_oof):.4f}")
print(f"Ridge Meta CV RMSE   : {rmse(y, ridge_oof):.4f}")
print(f"FINAL BLEND CV RMSE  : {final_cv_rmse:.4f} (Baseline: {baseline_rmse:.4f})")
print(f"FINAL BLEND CV Pearson: {final_cv_pear:.4f}\n")

# ==========================================
# 4. OPTIMAL LINEAR SHRINKAGE FOR RMSE
# ==========================================
calib = LinearRegression().fit(final_oof_blend.reshape(-1, 1), y)
a_scale = float(calib.coef_[0])
b_intercept = float(calib.intercept_)

calibrated_test_pred = np.clip(a_scale * final_test_blend + b_intercept, 0.0, 5.0)

print(f"Optimal Scale: a = {a_scale:.4f}, Intercept: b = {b_intercept:.4f}")
print(f"Calibrated Test Prediction Mean: {calibrated_test_pred.mean():.4f} (Train: {y.mean():.4f})")
print(f"Calibrated Test Prediction Std : {calibrated_test_pred.std():.4f}")
print(f"Calibrated Test Prediction Min/Max: [{calibrated_test_pred.min():.3f}, {calibrated_test_pred.max():.3f}]")

# ==========================================
# 5. GENERATE submission2.csv (PRESERVING submission.csv)
# ==========================================
sub2_df = pd.DataFrame({
    "filename": test_df[FILE_COL],
    "label": calibrated_test_pred
})

sub2_path = r"e:\SHL\submission2.csv"
sub2_df.to_csv(sub2_path, index=False)

print("\n" + "="*75)
print(f"SUCCESSFULLY SAVED NEW SUBMISSION TO: {sub2_path}")
print(f"Total Test Predictions: {len(sub2_df)}")
print("="*75)
print("\nFirst 10 rows of submission2.csv:")
print(sub2_df.head(10))
