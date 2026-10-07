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
from scipy.optimize import minimize

import torch
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold, KFold
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
# CONFIGURATION - TARGET: MINIMIZE RMSE
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = r"e:\SHL\data"
WORK_DIR = r"e:\SHL\work"

print("="*70)
print("     RMSE MINIMIZATION PIPELINE (LOWER IS BETTER)")
print("="*70)

# Load Data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
FILE_COL = "filename"
LABEL_COL = "label"

y = train_df[LABEL_COL].values.astype(float)
def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

baseline_rmse = np.sqrt(np.mean((y - y.mean())**2))
print(f"Target distribution: mean={y.mean():.4f}, std={y.std():.4f}")
print(f"Baseline Mean-Predictor RMSE: {baseline_rmse:.4f}\n")

# Load cached features
feat_cache = np.load(os.path.join(WORK_DIR, "all_features.npz"))
X_hand_tr = feat_cache["X_hand_tr"]
X_hand_te = feat_cache["X_hand_te"]
X_txt_tr  = feat_cache["X_txt_tr"]
X_txt_te  = feat_cache["X_txt_te"]
X_aud_tr  = feat_cache["X_aud_tr"]
X_aud_te  = feat_cache["X_aud_te"]

# Load prosody features if available
PROSODY_CACHE = os.path.join(WORK_DIR, "prosody_features.npz")
if os.path.exists(PROSODY_CACHE):
    p_data = np.load(PROSODY_CACHE)
    X_pro_tr = p_data["X_pro_tr"]
    X_pro_te = p_data["X_pro_te"]
    X_tab_tr = np.hstack([X_hand_tr, X_pro_tr])
    X_tab_te = np.hstack([X_hand_te, X_pro_te])
else:
    X_tab_tr = X_hand_tr
    X_tab_te = X_hand_te

print(f"Tabular features: {X_tab_tr.shape}")
print(f"Text features   : {X_txt_tr.shape}")
print(f"Audio features  : {X_aud_tr.shape}\n")

# ==========================================
# 1. PCA DECOMPOSITIONS & MULTI-VIEW FEATURE MATRICES
# ==========================================
scaler_aud = StandardScaler()
X_aud_scaled_tr = scaler_aud.fit_transform(X_aud_tr)
X_aud_scaled_te = scaler_aud.transform(X_aud_te)

pca_64 = PCA(n_components=64, random_state=SEED)
X_aud_pca64_tr = pca_64.fit_transform(X_aud_scaled_tr)
X_aud_pca64_te = pca_64.transform(X_aud_scaled_te)

pca_128 = PCA(n_components=128, random_state=SEED)
X_aud_pca128_tr = pca_128.fit_transform(X_aud_scaled_tr)
X_aud_pca128_te = pca_128.transform(X_aud_scaled_te)

# Concatenated representations
X_all_tr = np.hstack([StandardScaler().fit_transform(X_tab_tr), X_txt_tr, X_aud_pca128_tr])
X_all_te = np.hstack([StandardScaler().fit_transform(X_tab_te), X_txt_te, X_aud_pca128_te])

pca_all = PCA(n_components=96, random_state=SEED)
X_all_pca_tr = pca_all.fit_transform(X_all_tr)
X_all_pca_te = pca_all.transform(X_all_te)

# ==========================================
# 2. 10-FOLD STRATIFIED CROSS-VALIDATION
# ==========================================
N_FOLDS = 10
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
strat_bins = np.clip(np.round(y), 1, 5).astype(int)
folds = list(skf.split(np.zeros(len(y)), strat_bins))

alphas = np.logspace(-2, 4, 30)

MODELS = {
    # 1. Tabular tree boosting
    "tab_lgbm": (
        LGBMRegressor(n_estimators=180, learning_rate=0.03, num_leaves=12, max_depth=3, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0, random_state=SEED, verbose=-1),
        X_tab_tr, X_tab_te
    ),
    "tab_hgb": (
        HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=200, l2_regularization=2.0, random_state=SEED),
        X_tab_tr, X_tab_te
    ),
    "tab_ridge": (
        make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
        X_tab_tr, X_tab_te
    ),
    # 2. Text embeddings
    "txt_ridge": (
        make_pipeline(StandardScaler(), RidgeCV(alphas=alphas*10)),
        X_txt_tr, X_txt_te
    ),
    "txt_svr": (
        make_pipeline(StandardScaler(), SVR(C=1.5, epsilon=0.1, gamma="scale")),
        X_txt_tr, X_txt_te
    ),
    # 3. Audio representations
    "aud_ridge": (
        make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
        X_aud_pca128_tr, X_aud_pca128_te
    ),
    "aud_svr": (
        make_pipeline(StandardScaler(), SVR(C=1.5, epsilon=0.1, gamma="scale")),
        X_aud_pca64_tr, X_aud_pca64_te
    ),
    "aud_lgbm": (
        LGBMRegressor(n_estimators=150, learning_rate=0.03, num_leaves=10, max_depth=3, subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=3.0, random_state=SEED, verbose=-1),
        X_aud_pca64_tr, X_aud_pca64_te
    ),
    # 4. Multi-view concatenated
    "all_ridge": (
        make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
        X_all_pca_tr, X_all_pca_te
    ),
    "all_huber": (
        make_pipeline(StandardScaler(), HuberRegressor(max_iter=300, alpha=50.0)),
        X_all_pca_tr, X_all_pca_te
    ),
    "all_hgb": (
        HistGradientBoostingRegressor(max_depth=3, learning_rate=0.02, max_iter=150, l2_regularization=5.0, random_state=SEED),
        X_all_pca_tr, X_all_pca_te
    )
}

print("--- 2. Training Individual Models (10-Fold CV) ---")
oof_preds = {}
test_preds = {}

for name, (model, Xtr, Xte) in MODELS.items():
    p_oof = np.zeros(len(y))
    p_test_folds = np.zeros(len(test_df))
    
    for tr_idx, val_idx in folds:
        m = model.fit(Xtr[tr_idx], y[tr_idx])
        p_oof[val_idx] = m.predict(Xtr[val_idx])
        p_test_folds += m.predict(Xte) / N_FOLDS
        
    p_oof = np.clip(p_oof, 0.0, 5.0)
    p_test_folds = np.clip(p_test_folds, 0.0, 5.0)
    
    oof_preds[name] = p_oof
    test_preds[name] = p_test_folds
    
    m_rmse = rmse(y, p_oof)
    m_pear = pearsonr(y, p_oof)[0]
    print(f"Model: {name:12s} | 10-Fold OOF RMSE: {m_rmse:.4f} (Pearson: {m_pear:.4f})")

# ==========================================
# 3. NON-NEGATIVE LEAST SQUARES STACKING (OPTIMAL RMSE MINIMIZATION)
# ==========================================
print("\n" + "="*70)
print("3. STACKING META-REGRESSOR (STRICT RMSE MINIMIZATION)")
print("="*70)

OOF_MAT = np.column_stack([oof_preds[k] for k in MODELS])
TEST_MAT = np.column_stack([test_preds[k] for k in MODELS])

# Non-Negative Linear Regression strictly minimizes ||y - Xw||_2^2
meta = LinearRegression(positive=True)

# Out-of-fold cross-validated prediction for the stacker itself
cv_stack_oof = np.zeros(len(y))
for tr_idx, val_idx in folds:
    m_fold = LinearRegression(positive=True).fit(OOF_MAT[tr_idx], y[tr_idx])
    cv_stack_oof[val_idx] = m_fold.predict(OOF_MAT[val_idx])

cv_stack_oof = np.clip(cv_stack_oof, 0.0, 5.0)
meta.fit(OOF_MAT, y)

test_pred_raw = np.clip(meta.predict(TEST_MAT), 0.0, 5.0)

stack_rmse = rmse(y, cv_stack_oof)
stack_pear = pearsonr(y, cv_stack_oof)[0]

print(f"Active Stacker Weights:\n{dict(zip(MODELS.keys(), meta.coef_.round(4)))}\n")
print(f"Stacker Intercept : {meta.intercept_:.4f}")
print(f"STACKER CV RMSE   : {stack_rmse:.4f}  (Baseline: {baseline_rmse:.4f})")
print(f"STACKER CV Pearson: {stack_pear:.4f}\n")

# ==========================================
# 4. OPTIMAL POST-PROCESSING CALIBRATION FOR RMSE
# ==========================================
# In RMSE, the optimal shrinkage/bias correction minimizes MSE directly:
# y_hat_opt = a * y_hat + b
opt_calib = LinearRegression().fit(cv_stack_oof.reshape(-1, 1), y)
a_scale = float(opt_calib.coef_[0])
b_intercept = float(opt_calib.intercept_)

final_test_pred = np.clip(a_scale * test_pred_raw + b_intercept, 0.0, 5.0)

print(f"Optimal RMSE Calibration: a = {a_scale:.4f}, b = {b_intercept:.4f}")
print(f"Final Test Prediction Range: [{final_test_pred.min():.3f}, {final_test_pred.max():.3f}]")
print(f"Final Test Prediction Mean : {final_test_pred.mean():.3f} (Train Mean: {y.mean():.3f})")
print(f"Final Test Prediction Std  : {final_test_pred.std():.3f}")

# ==========================================
# 5. GENERATE SUBMISSION FILE
# ==========================================
sub_df = pd.DataFrame({
    "filename": test_df[FILE_COL],
    "label": final_test_pred
})

sub_path = r"e:\SHL\submission.csv"
sub_df.to_csv(sub_path, index=False)

print("\n" + "="*70)
print(f"SAVED OPTIMAL RMSE SUBMISSION TO: {sub_path}")
print(f"Total Rows: {len(sub_df)}")
print("="*70)
print("\nFirst 10 submission rows:")
print(sub_df.head(10))
