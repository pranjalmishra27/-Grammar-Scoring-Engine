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
from scipy.stats import pearsonr, skew
from scipy.optimize import minimize

import torch
import librosa
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV, HuberRegressor, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = r"e:\SHL\data"
WORK_DIR = r"e:\SHL\work"
SR = 16000

print("="*70)
print("     ADVANCED HIGH-PERFORMANCE GRAMMAR SCORING ENGINE (V2)")
print("="*70)

# 1. Load Data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
FILE_COL = "filename"
LABEL_COL = "label"

y = train_df[LABEL_COL].values.astype(float)
print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")
print(f"Target distribution: mean={y.mean():.3f}, std={y.std():.3f}, min={y.min():.1f}, max={y.max():.1f}")

# Load cached transcripts
TR_CACHE = os.path.join(WORK_DIR, "transcripts.json")
assert os.path.exists(TR_CACHE), "transcripts.json not found in work directory!"
transcripts = json.load(open(TR_CACHE, "r", encoding="utf-8"))
print(f"Loaded {len(transcripts)} cached transcripts.")

train_df["text"] = [transcripts[f"train_{fn}"]["text"] for fn in train_df[FILE_COL]]
test_df["text"]  = [transcripts[f"test_{fn}"]["text"]  for fn in test_df[FILE_COL]]
train_df["seg"]  = [transcripts[f"train_{fn}"]["seg"]  for fn in train_df[FILE_COL]]
test_df["seg"]   = [transcripts[f"test_{fn}"]["seg"]   for fn in test_df[FILE_COL]]

# Load cached wav2vec2 & text embeddings
feat_cache = np.load(os.path.join(WORK_DIR, "all_features.npz"))
X_txt_tr = feat_cache["X_txt_tr"]
X_txt_te = feat_cache["X_txt_te"]
X_aud_tr = feat_cache["X_aud_tr"]
X_aud_te = feat_cache["X_aud_te"]
print(f"Loaded Text Embeddings: {X_txt_tr.shape} | Wav2Vec2 Embeddings: {X_aud_tr.shape}\n")

# ==========================================
# 2. EXTRACT RICH ACOUSTIC PROSODIC FEATURES
# ==========================================
PROSODY_CACHE = os.path.join(WORK_DIR, "prosody_features.npz")

if os.path.exists(PROSODY_CACHE):
    print("Loading cached acoustic prosodic features...")
    p_data = np.load(PROSODY_CACHE)
    X_pro_tr = p_data["X_pro_tr"]
    X_pro_te = p_data["X_pro_te"]
    pro_cols = p_data["cols"].tolist()
else:
    print("Extracting acoustic prosody & acoustic feature dynamics from audio...")
    def extract_prosody_features(wav_path):
        wav, _ = librosa.load(wav_path, sr=SR, mono=True)
        wav, _ = librosa.effects.trim(wav, top_db=30)
        dur = len(wav) / SR
        
        # Energy & RMS
        rms = librosa.feature.rms(y=wav)[0]
        rms_mean = float(np.mean(rms))
        rms_std  = float(np.std(rms))
        rms_max  = float(np.max(rms))
        rms_dyn  = float(rms_max - np.min(rms))
        
        # Zero-Crossing Rate (speech vs silence / unvoiced consonants)
        zcr = librosa.feature.zero_crossing_rate(wav)[0]
        zcr_mean = float(np.mean(zcr))
        zcr_std  = float(np.std(zcr))
        
        # Spectral properties
        sc = librosa.feature.spectral_centroid(y=wav, sr=SR)[0]
        sc_mean = float(np.mean(sc))
        sc_std  = float(np.std(sc))
        
        sr_roll = librosa.feature.spectral_rolloff(y=wav, sr=SR)[0]
        sr_mean = float(np.mean(sr_roll))
        sr_std  = float(np.std(sr_roll))
        
        sf = librosa.feature.spectral_flatness(y=wav)[0]
        sf_mean = float(np.mean(sf))
        sf_std  = float(np.std(sf))
        
        # 20 MFCCs (mean, std, skew)
        mfcc = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=20)
        mfcc_means = np.mean(mfcc, axis=1)
        mfcc_stds  = np.std(mfcc, axis=1)
        mfcc_skews = skew(mfcc, axis=1)
        
        feats = {
            "dur": dur,
            "rms_mean": rms_mean, "rms_std": rms_std, "rms_max": rms_max, "rms_dyn": rms_dyn,
            "zcr_mean": zcr_mean, "zcr_std": zcr_std,
            "sc_mean": sc_mean, "sc_std": sc_std,
            "sr_mean": sr_mean, "sr_std": sr_std,
            "sf_mean": sf_mean, "sf_std": sf_std,
        }
        for i in range(20):
            feats[f"mfcc_m_{i}"] = float(mfcc_means[i])
            feats[f"mfcc_s_{i}"] = float(mfcc_stds[i])
            feats[f"mfcc_k_{i}"] = float(mfcc_skews[i])
            
        return feats

    train_pros = [extract_prosody_features(os.path.join(DATA_DIR, "train", fn)) for fn in tqdm(train_df[FILE_COL], desc="Train Prosody")]
    test_pros  = [extract_prosody_features(os.path.join(DATA_DIR, "test", fn))  for fn in tqdm(test_df[FILE_COL],  desc="Test Prosody")]
    
    df_pro_tr = pd.DataFrame(train_pros)
    df_pro_te = pd.DataFrame(test_pros)
    X_pro_tr = df_pro_tr.values
    X_pro_te = df_pro_te.values
    pro_cols = df_pro_tr.columns.tolist()
    
    np.savez_compressed(PROSODY_CACHE, X_pro_tr=X_pro_tr, X_pro_te=X_pro_te, cols=np.array(pro_cols))

print(f"Prosodic Feature Matrix: {X_pro_tr.shape}")

# ==========================================
# 3. ADVANCED LINGUISTIC & SYNTACTIC FEATURES
# ==========================================
print("\nExtracting rich syntactic, fluency, and discourse features...")
FILLERS = {"um", "uh", "uhm", "erm", "hmm", "like", "basically", "actually", "so", "you know", "i mean", "sort of", "kind of"}
SUBORDINATING = {"because", "although", "which", "while", "whereas", "however", "if", "when", "that", "since", "unless", "despite", "whereas"}
COORDINATING = {"and", "but", "or", "so", "for", "yet", "nor"}
TRANSITIONS = {"moreover", "furthermore", "therefore", "additionally", "nevertheless", "in fact", "as a result", "consequently"}
AGREEMENT_PATTERNS = [
    r"\bhe are\b", r"\bshe are\b", r"\bit are\b", r"\bthey is\b", r"\bwe is\b", r"\byou is\b",
    r"\bhe have\b", r"\bshe have\b", r"\bit have\b", r"\bi is\b", r"\bwas were\b", r"\bdo does\b",
    r"\bthese is\b", r"\bthose is\b", r"\bthis are\b", r"\bthat are\b"
]

def extract_advanced_linguistics(text, seg, dur):
    words = re.findall(r"[A-Za-z']+", text.lower())
    n_words = max(len(words), 1)
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    slen = [len(re.findall(r"[A-Za-z']+", s)) for s in sents] or [0]
    
    # Repetitions & false starts
    rep_1 = sum(1 for a, b in zip(words, words[1:]) if a == b)
    rep_2 = sum(1 for i in range(len(words)-3) if words[i:i+2] == words[i+2:i+4])
    
    # Timing & Speech Dynamics
    speech_time = sum(s[1] - s[0] for s in seg) if seg else dur
    gaps = [seg[i+1][0] - seg[i][1] for i in range(len(seg)-1)] if len(seg) > 1 else [0.0]
    
    # Lexical Diversity
    vocab = set(words)
    ttr = len(vocab) / n_words
    root_ttr = len(vocab) / np.sqrt(n_words)
    bilog_ttr = np.log(max(len(vocab), 1)) / np.log(max(n_words, 2))
    
    # Connectives & Complexity
    sub_count = sum(w in SUBORDINATING for w in words)
    coor_count = sum(w in COORDINATING for w in words)
    trans_count = sum(w in TRANSITIONS for w in words)
    filler_count = sum(w in FILLERS for w in words)
    aggr_errors = sum(len(re.findall(pat, text.lower())) for pat in AGREEMENT_PATTERNS)
    
    return {
        "n_words": len(words),
        "n_sents": len(sents),
        "wps_total": len(words) / max(dur, 1e-3),
        "wps_speech": len(words) / max(speech_time, 1e-3),
        "speech_ratio": speech_time / max(dur, 1e-3),
        "ttr": ttr,
        "root_ttr": root_ttr,
        "bilog_ttr": bilog_ttr,
        "avg_wlen": np.mean([len(w) for w in words]) if words else 0.0,
        "avg_slen": np.mean(slen),
        "max_slen": np.max(slen),
        "std_slen": np.std(slen),
        "frac_short_sents": np.mean([l < 4 for l in slen]),
        "frac_long_sents": np.mean([l > 15 for l in slen]),
        "filler_rate": filler_count / n_words,
        "repeat_1_rate": rep_1 / n_words,
        "repeat_2_rate": rep_2 / n_words,
        "sub_conj_rate": sub_count / n_words,
        "coor_conj_rate": coor_count / n_words,
        "trans_rate": trans_count / n_words,
        "agreement_err_rate": aggr_errors / n_words,
        "gap_mean": np.mean(gaps),
        "gap_max": np.max(gaps),
        "gap_std": np.std(gaps),
        "long_word_rate": np.mean([len(w) > 6 for w in words]) if words else 0.0,
        "comma_rate": text.count(",") / n_words,
        "dur": dur,
    }

# Get duration from prosody
dur_tr = X_pro_tr[:, 0]
dur_te = X_pro_te[:, 0]

df_ling_tr = pd.DataFrame([extract_advanced_linguistics(t, s, d) for t, s, d in zip(train_df.text, train_df.seg, dur_tr)])
df_ling_te = pd.DataFrame([extract_advanced_linguistics(t, s, d) for t, s, d in zip(test_df.text,  test_df.seg,  dur_te)])

# Merge Tabular Features (Linguistic + Prosodic)
X_tab_tr = np.hstack([df_ling_tr.values, X_pro_tr])
X_tab_te = np.hstack([df_ling_te.values, X_pro_te])
print(f"Complete Tabular Feature Matrix: train={X_tab_tr.shape}, test={X_tab_te.shape}\n")

# ==========================================
# 4. CROSS-VALIDATION & MULTI-MODEL ENSEMBLE
# ==========================================
print("="*70)
print("4. 10-FOLD STRATIFIED CROSS-VALIDATION & MODEL BENCHMARKING")
print("="*70)

def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

N_FOLDS = 10
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
strat_bins = np.clip(np.round(y), 1, 5).astype(int)
folds = list(skf.split(np.zeros(len(y)), strat_bins))

# Prepare PCA representations
pca_aud_64 = PCA(n_components=64, random_state=SEED)
X_aud_pca64_tr = pca_aud_64.fit_transform(StandardScaler().fit_transform(X_aud_tr))
X_aud_pca64_te = pca_aud_64.transform(StandardScaler().fit_transform(X_aud_te))

pca_aud_128 = PCA(n_components=128, random_state=SEED)
X_aud_pca128_tr = pca_aud_128.fit_transform(StandardScaler().fit_transform(X_aud_tr))
X_aud_pca128_te = pca_aud_128.transform(StandardScaler().fit_transform(X_aud_te))

# Concatenated full view
X_all_raw_tr = np.hstack([X_tab_tr, X_txt_tr, X_aud_pca128_tr])
X_all_raw_te = np.hstack([X_tab_te, X_txt_te, X_aud_pca128_te])

pca_all = PCA(n_components=128, random_state=SEED)
X_all_pca_tr = pca_all.fit_transform(StandardScaler().fit_transform(X_all_raw_tr))
X_all_pca_te = pca_all.transform(StandardScaler().fit_transform(X_all_raw_te))

alphas = np.logspace(-2, 4, 30)

MODELS = {
    "tab_lgbm": (
        LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=15, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1),
        X_tab_tr, X_tab_te
    ),
    "tab_hgb": (
        HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=250, l2_regularization=1.5, random_state=SEED),
        X_tab_tr, X_tab_te
    ),
    "tab_et": (
        make_pipeline(StandardScaler(), ExtraTreesRegressor(n_estimators=200, max_depth=8, min_samples_split=4, random_state=SEED, n_jobs=-1)),
        X_tab_tr, X_tab_te
    ),
    "tab_rf": (
        make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200, max_depth=7, min_samples_split=4, random_state=SEED, n_jobs=-1)),
        X_tab_tr, X_tab_te
    ),
    "tab_ridge": (
        make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
        X_tab_tr, X_tab_te
    ),
    "txt_ridge": (
        make_pipeline(StandardScaler(), RidgeCV(alphas=alphas*10)),
        X_txt_tr, X_txt_te
    ),
    "txt_svr": (
        make_pipeline(StandardScaler(), SVR(C=2.5, epsilon=0.08, gamma="scale")),
        X_txt_tr, X_txt_te
    ),
    "aud_svr": (
        make_pipeline(StandardScaler(), SVR(C=2.5, epsilon=0.08, gamma="scale")),
        X_aud_pca64_tr, X_aud_pca64_te
    ),
    "aud_ridge": (
        make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
        X_aud_pca128_tr, X_aud_pca128_te
    ),
    "aud_lgbm": (
        LGBMRegressor(n_estimators=180, learning_rate=0.03, num_leaves=12, max_depth=3, subsample=0.8, colsample_bytree=0.7, random_state=SEED, verbose=-1),
        X_aud_pca64_tr, X_aud_pca64_te
    ),
    "all_ridge": (
        make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
        X_all_pca_tr, X_all_pca_te
    ),
    "all_huber": (
        make_pipeline(RobustScaler(), HuberRegressor(max_iter=300, alpha=10.0)),
        X_all_pca_tr, X_all_pca_te
    ),
}

oof_preds = {}
test_preds = {}

for name, (model, Xtr, Xte) in MODELS.items():
    p_oof = np.zeros(len(y))
    p_test_folds = np.zeros(len(test_df))
    
    for tr_idx, val_idx in folds:
        # Fit on fold
        m = model.fit(Xtr[tr_idx], y[tr_idx])
        p_oof[val_idx] = m.predict(Xtr[val_idx])
        p_test_folds += m.predict(Xte) / N_FOLDS
        
    p_oof = np.clip(p_oof, 0.0, 5.0)
    p_test_folds = np.clip(p_test_folds, 0.0, 5.0)
    
    oof_preds[name] = p_oof
    test_preds[name] = p_test_folds
    
    m_rmse = rmse(y, p_oof)
    m_pear = pearsonr(y, p_oof)[0]
    print(f"Model: {name:12s} | 10-Fold OOF RMSE: {m_rmse:.4f} | Pearson r: {m_pear:.4f}")

# ==========================================
# 5. DIRECT PEARSON-MAXIMIZING META ENSEMBLE
# ==========================================
print("\n" + "="*70)
print("5. DIRECT PEARSON CORRELATION META-OPTIMIZATION")
print("="*70)

OOF_MAT = np.column_stack([oof_preds[k] for k in MODELS])
TEST_MAT = np.column_stack([test_preds[k] for k in MODELS])
num_models = OOF_MAT.shape[1]

# Objective: Minimize negative Pearson correlation
def neg_pearson_objective(weights):
    weights = np.array(weights)
    pred = OOF_MAT @ weights
    r, _ = pearsonr(y, pred)
    return -r

init_weights = np.ones(num_models) / num_models
bounds = [(0.0, 1.0) for _ in range(num_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    neg_pearson_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 1000, "ftol": 1e-8}
)

best_weights = opt_res.x
best_weights = np.maximum(best_weights, 0.0)
best_weights /= np.sum(best_weights)

weight_dict = {k: round(float(w), 4) for k, w in zip(MODELS.keys(), best_weights) if w > 0.005}
print(f"Optimized Active Ensemble Weights:\n{weight_dict}\n")

ensemble_oof = OOF_MAT @ best_weights
ensemble_test = TEST_MAT @ best_weights

opt_rmse = rmse(y, ensemble_oof)
opt_pear = pearsonr(y, ensemble_oof)[0]

print(f"ENSEMBLE 10-Fold CV RMSE       : {opt_rmse:.4f}")
print(f"ENSEMBLE 10-Fold CV Pearson (r): {opt_pear:.4f}")

# ==========================================
# 6. EMPIRICAL DISTRIBUTION CALIBRATION
# ==========================================
print("\n--- Calibrating Test Predictions to Empirical Train Distribution ---")

# Match mean and standard deviation of predictions to true train distribution
train_mean, train_std = np.mean(y), np.std(y)
pred_mean, pred_std   = np.mean(ensemble_test), np.std(ensemble_test)

calibrated_test = (ensemble_test - pred_mean) * (train_std / max(pred_std, 1e-4)) + train_mean
calibrated_test = np.clip(calibrated_test, 1.0, 5.0)

# Blending standard and calibrated predictions (80% calibrated, 20% raw for optimal stability)
final_test_pred = 0.85 * calibrated_test + 0.15 * np.clip(ensemble_test, 1.0, 5.0)

print(f"Raw Test Mean: {ensemble_test.mean():.3f}, Std: {ensemble_test.std():.3f}")
print(f"Calibrated Test Mean: {final_test_pred.mean():.3f}, Std: {final_test_pred.std():.3f}")
print(f"Prediction Range: [{final_test_pred.min():.3f}, {final_test_pred.max():.3f}]")

# ==========================================
# 7. GENERATE FINAL SUBMISSION FILE
# ==========================================
sub_df = pd.DataFrame({
    "filename": test_df[FILE_COL],
    "label": final_test_pred
})

sub_path = r"e:\SHL\submission.csv"
sub_df.to_csv(sub_path, index=False)

print("\n" + "="*70)
print(f"SAVED FINAL ENSEMBLE SUBMISSION TO: {sub_path}")
print(f"Total Rows: {len(sub_df)}")
print("="*70)
print("\nFirst 10 submission rows:")
print(sub_df.head(10))
