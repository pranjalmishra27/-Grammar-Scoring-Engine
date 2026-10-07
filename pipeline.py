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
import librosa
from faster_whisper import WhisperModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_SIZE = "tiny.en" if DEVICE == "cpu" else "small.en"

os.makedirs(WORK_DIR, exist_ok=True)
print("="*65)
print("     SHL HIRING ASSESSMENT - GRAMMAR SCORING ENGINE")
print("="*65)
print(f"Device        : {DEVICE}")
print(f"Whisper Model : {WHISPER_SIZE}")
print(f"Data Directory: {DATA_DIR}")
print(f"Work Directory: {WORK_DIR}\n")

# ==========================================
# 1. LOAD CSVs & AUDIO PATHS
# ==========================================
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub_df   = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

FILE_COL  = train_df.columns[0]
LABEL_COL = [c for c in train_df.columns if c != FILE_COL][0]

print(f"Loaded Train Data : {train_df.shape} (File: '{FILE_COL}', Label: '{LABEL_COL}')")
print(f"Loaded Test Data  : {test_df.shape}")
print(f"Sample Sub Shape  : {sub_df.shape}")

train_audio_paths = [os.path.join(DATA_DIR, "train", fn) for fn in train_df[FILE_COL]]
test_audio_paths  = [os.path.join(DATA_DIR, "test", fn)  for fn in test_df[FILE_COL]]

# Verify paths exist
assert all(os.path.exists(p) for p in train_audio_paths), "Missing some train audio files!"
assert all(os.path.exists(p) for p in test_audio_paths), "Missing some test audio files!"
print(f"Verified all 769 train audio and 216 test audio files on disk.\n")

# ==========================================
# 2. AUDIO PREPROCESSING (Trim & Normalise)
# ==========================================
def load_audio(path):
    wav, _ = librosa.load(path, sr=SR, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    peak = np.max(np.abs(wav)) + 1e-8
    return (wav / peak).astype(np.float32)

print("--- 2. Loading & Preprocessing Audio Clips ---")
train_wavs = [load_audio(p) for p in tqdm(train_audio_paths, desc="Preprocessing train wavs")]
test_wavs  = [load_audio(p) for p in tqdm(test_audio_paths,  desc="Preprocessing test wavs")]

train_df["dur"] = [len(w) / SR for w in train_wavs]
test_df["dur"]  = [len(w) / SR for w in test_wavs]
print(f"Duration (Train): mean={train_df['dur'].mean():.2f}s, min={train_df['dur'].min():.2f}s, max={train_df['dur'].max():.2f}s\n")

# ==========================================
# 3. ASR TRANSCRIPTION (faster-whisper)
# ==========================================
TR_CACHE = os.path.join(WORK_DIR, "transcripts.json")
cache = json.load(open(TR_CACHE, "r", encoding="utf-8")) if os.path.exists(TR_CACHE) else {}

print(f"--- 3. Speech-to-Text Transcription (faster-whisper) ---")
print(f"Found {len(cache)} cached transcripts.")

asr_model = WhisperModel(WHISPER_SIZE, device=DEVICE, compute_type="int8" if DEVICE == "cpu" else "float16", num_workers=4)

def transcribe_clip(key, wav):
    if key in cache:
        return cache[key]
    segs, _ = asr_model.transcribe(wav, language="en", beam_size=1, vad_filter=True, condition_on_previous_text=False)
    segs = list(segs)
    cache[key] = {
        "text": " ".join(s.text.strip() for s in segs),
        "seg": [(float(s.start), float(s.end)) for s in segs]
    }
    return cache[key]

# Transcribe train
for fn, wav in tqdm(list(zip(train_df[FILE_COL], train_wavs)), desc="Transcribing Train"):
    key = f"train_{fn}"
    transcribe_clip(key, wav)

# Transcribe test
for fn, wav in tqdm(list(zip(test_df[FILE_COL], test_wavs)), desc="Transcribing Test"):
    key = f"test_{fn}"
    transcribe_clip(key, wav)

with open(TR_CACHE, "w", encoding="utf-8") as f:
    json.dump(cache, f)

del asr_model
if DEVICE == "cuda":
    torch.cuda.empty_cache()

train_df["text"] = [cache[f"train_{fn}"]["text"] for fn in train_df[FILE_COL]]
test_df["text"]  = [cache[f"test_{fn}"]["text"]  for fn in test_df[FILE_COL]]
train_df["seg"]  = [cache[f"train_{fn}"]["seg"]  for fn in train_df[FILE_COL]]
test_df["seg"]   = [cache[f"test_{fn}"]["seg"]   for fn in test_df[FILE_COL]]

print(f"Sample Train Transcript [0] (Score {train_df[LABEL_COL].iloc[0]}):")
print(f"  \"{train_df['text'].iloc[0][:200]}...\"\n")

# ==========================================
# 4. FEATURE EXTRACTION
# ==========================================
print("--- 4. Feature Extraction ---")
FILLERS = {"um", "uh", "uhm", "erm", "hmm", "like", "basically", "actually", "so", "you know", "i mean"}
CONJUNCTIONS = {"because", "although", "which", "while", "whereas", "however", "if", "when", "that", "since", "therefore", "moreover", "unless", "despite", "furthermore"}
COMMON_AGREEMENT_ERRORS = [
    r"\bhe are\b", r"\bshe are\b", r"\bit are\b", r"\bthey is\b", r"\bwe is\b", r"\byou is\b",
    r"\bhe have\b", r"\bshe have\b", r"\bit have\b", r"\bi is\b", r"\bwas were\b", r"\bdo does\b"
]

def extract_handcrafted_features(text, dur, seg):
    words = re.findall(r"[A-Za-z']+", text.lower())
    n_words = max(len(words), 1)
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    slen = [len(re.findall(r"[A-Za-z']+", s)) for s in sents] or [0]
    rep = sum(1 for a, b in zip(words, words[1:]) if a == b)
    gaps = [seg[i+1][0] - seg[i][1] for i in range(len(seg)-1)] if len(seg) > 1 else [0.0]
    speech_time = sum(s[1] - s[0] for s in seg) if seg else dur
    agreement_errs = sum(len(re.findall(pat, text.lower())) for pat in COMMON_AGREEMENT_ERRORS)

    return {
        "n_words": len(words),
        "n_sents": len(sents),
        "wps_total": len(words) / max(dur, 1e-3),
        "wps_speech": len(words) / max(speech_time, 1e-3),
        "ttr": len(set(words)) / n_words,
        "root_ttr": len(set(words)) / np.sqrt(n_words),
        "avg_wlen": np.mean([len(w) for w in words]) if words else 0.0,
        "avg_slen": np.mean(slen),
        "max_slen": np.max(slen),
        "std_slen": np.std(slen),
        "frac_short_sents": np.mean([l < 4 for l in slen]),
        "filler_rate": sum(w in FILLERS for w in words) / n_words,
        "repeat_rate": rep / n_words,
        "agreement_err_rate": agreement_errs / n_words,
        "n_segs": len(seg),
        "speech_ratio": speech_time / max(dur, 1e-3),
        "gap_mean": np.mean(gaps),
        "gap_max": np.max(gaps),
        "gap_std": np.std(gaps),
        "long_word_rate": np.mean([len(w) > 6 for w in words]) if words else 0.0,
        "comma_rate": text.count(",") / n_words,
        "conj_rate": sum(w in CONJUNCTIONS for w in words) / n_words,
        "dur": dur,
    }

Xb_tr = pd.DataFrame([extract_handcrafted_features(t, d, s) for t, d, s in zip(train_df.text, train_df.dur, train_df.seg)])
Xb_te = pd.DataFrame([extract_handcrafted_features(t, d, s) for t, d, s in zip(test_df.text,  test_df.dur,  test_df.seg)])

# 4b. GPT-2 Surprisal & Perplexity
print("--> Extracting GPT-2 Surprisal & Perplexity...")
tok = GPT2TokenizerFast.from_pretrained("gpt2")
lm = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()

@torch.no_grad()
def lm_feats(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(DEVICE)
    if ids.shape[1] < 3:
        return dict(lm_mean=0.0, lm_std=0.0, lm_p90=0.0, lm_max=0.0, lm_hi=0.0, ppl=0.0)
    logits = lm(ids).logits[:, :-1]
    loss = torch.nn.functional.cross_entropy(logits[0], ids[0, 1:], reduction="none").cpu().numpy()
    return dict(
        lm_mean=float(loss.mean()),
        lm_std=float(loss.std()),
        lm_p90=float(np.percentile(loss, 90)),
        lm_max=float(loss.max()),
        lm_hi=float((loss > 6.0).mean()),
        ppl=float(np.exp(np.clip(loss.mean(), 0, 15)))
    )

Xl_tr = pd.DataFrame([lm_feats(t) for t in tqdm(train_df.text, desc="GPT-2 Train")])
Xl_te = pd.DataFrame([lm_feats(t) for t in tqdm(test_df.text,  desc="GPT-2 Test")])
del lm
if DEVICE == "cuda":
    torch.cuda.empty_cache()

X_hand_tr = pd.concat([Xb_tr, Xl_tr], axis=1).fillna(0)
X_hand_te = pd.concat([Xb_te, Xl_te], axis=1).fillna(0)
print(f"Handcrafted + Surprisal features: {X_hand_tr.shape}")

# 4c. Sentence-Transformers Semantic & Quality Embeddings
print("--> Extracting Sentence Transformer Embeddings...")
st_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
X_txt_tr = st_model.encode(train_df.text.tolist(), batch_size=32, show_progress_bar=True, normalize_embeddings=True)
X_txt_te = st_model.encode(test_df.text.tolist(),  batch_size=32, show_progress_bar=True, normalize_embeddings=True)
del st_model
print(f"Text embeddings shape: train={X_txt_tr.shape}, test={X_txt_te.shape}")

# 4d. Wav2Vec2 Acoustic Embeddings
print("--> Extracting Wav2Vec2 Acoustic Embeddings...")
AUD_CACHE = os.path.join(WORK_DIR, "wav2vec2_feats.npz")
if os.path.exists(AUD_CACHE):
    cached_aud = np.load(AUD_CACHE)
    X_aud_tr = cached_aud["X_aud_tr"]
    X_aud_te = cached_aud["X_aud_te"]
    print(f"Loaded cached Wav2Vec2 features: {X_aud_tr.shape}")
else:
    fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE).eval()

    @torch.no_grad()
    def audio_emb(wav):
        x = fe(wav, sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE)
        h = w2v(x).last_hidden_state[0]
        return torch.cat([h.mean(0), h.std(0)]).cpu().numpy()

    X_aud_tr = np.stack([audio_emb(w) for w in tqdm(train_wavs, desc="wav2vec2 train")])
    X_aud_te = np.stack([audio_emb(w) for w in tqdm(test_wavs,  desc="wav2vec2 test")])
    np.savez_compressed(AUD_CACHE, X_aud_tr=X_aud_tr, X_aud_te=X_aud_te)
    del w2v
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print(f"Audio embeddings shape: train={X_aud_tr.shape}, test={X_aud_te.shape}")

# Save all feature matrices
np.savez_compressed(
    os.path.join(WORK_DIR, "all_features.npz"),
    X_hand_tr=X_hand_tr.values, X_hand_te=X_hand_te.values,
    X_txt_tr=X_txt_tr, X_txt_te=X_txt_te,
    X_aud_tr=X_aud_tr, X_aud_te=X_aud_te,
    y=train_df[LABEL_COL].values
)
print("Saved all extracted features to work/all_features.npz\n")

# ==========================================
# 5. CROSS-VALIDATION & MODEL TRAINING
# ==========================================
print("="*65)
print("5. MODEL TRAINING & 5-FOLD STRATIFIED CV EVALUATION")
print("="*65)

y = train_df[LABEL_COL].values.astype(float)
def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
strat = np.clip(np.round(y), 1, 5).astype(int)
folds = list(skf.split(np.zeros(len(y)), strat))

Xh_tr, Xh_te = X_hand_tr.values, X_hand_te.values
X_all_tr = np.hstack([Xh_tr, X_txt_tr, X_aud_tr])
X_all_te = np.hstack([Xh_te, X_txt_te, X_aud_te])

alphas = np.logspace(-2, 4, 25)

def block_pipe(kind):
    if kind == "hand_hgb":
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=SEED)
    if kind == "hand_ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    if kind == "txt_ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas*10))
    if kind == "txt_svr":
        return make_pipeline(StandardScaler(), SVR(C=2.0, epsilon=0.1, gamma="scale"))
    if kind == "aud_svr":
        return make_pipeline(StandardScaler(), PCA(64, random_state=SEED), SVR(C=2.0, epsilon=0.1, gamma="scale"))
    if kind == "all_ridge":
        return make_pipeline(StandardScaler(), PCA(128, random_state=SEED), RidgeCV(alphas=alphas))
    raise ValueError(kind)

SPEC = {
    "hand_hgb":   ("hand_hgb",   Xh_tr,      Xh_te),
    "hand_ridge": ("hand_ridge", Xh_tr,      Xh_te),
    "txt_ridge":  ("txt_ridge",  X_txt_tr,   X_txt_te),
    "txt_svr":    ("txt_svr",    X_txt_tr,   X_txt_te),
    "aud_svr":    ("aud_svr",    X_aud_tr,   X_aud_te),
    "all_ridge":  ("all_ridge",  X_all_tr,   X_all_te),
}

oof = {}
for name, (kind, Xtr, _) in SPEC.items():
    p = np.zeros(len(y))
    for tr_i, va_i in folds:
        m = block_pipe(kind).fit(Xtr[tr_i], y[tr_i])
        p[va_i] = m.predict(Xtr[va_i])
    oof[name] = np.clip(p, 0.0, 5.0)
    score_rmse = rmse(y, oof[name])
    score_pearson = pearsonr(y, oof[name])[0]
    print(f"Model: {name:12s} | OOF RMSE: {score_rmse:.4f} | Pearson Correlation: {score_pearson:.4f}")

# ==========================================
# 6. NON-NEGATIVE LINEAR STACKING
# ==========================================
print("\n--- 6. Training Non-Negative Stacking Meta-Regressor ---")
O = np.column_stack([oof[k] for k in SPEC])
meta = LinearRegression(positive=True)
stack_oof = np.clip(cross_val_predict(meta, O, y, cv=skf.split(O, strat)), 0.0, 5.0)
meta.fit(O, y)

print(f"Stacker Weights   : {dict(zip(SPEC.keys(), meta.coef_.round(4)))}")
print(f"Stacker Intercept : {meta.intercept_:.4f}")
print(f"STACKER CV RMSE   : {rmse(y, stack_oof):.4f}")
print(f"STACKER CV Pearson: {pearsonr(y, stack_oof)[0]:.4f}")

# ==========================================
# 7. FINAL FIT & COMPREHENSIVE PERFORMANCE REPORT
# ==========================================
print("\n" + "="*65)
print("7. FINAL FIT & EVALUATION SUMMARY")
print("="*65)

train_pred_base, test_pred_base = {}, {}
for name, (kind, Xtr, Xte) in SPEC.items():
    m = block_pipe(kind).fit(Xtr, y)
    train_pred_base[name] = np.clip(m.predict(Xtr), 0.0, 5.0)
    test_pred_base[name]  = np.clip(m.predict(Xte), 0.0, 5.0)

Ttr = np.column_stack([train_pred_base[k] for k in SPEC])
Tte = np.column_stack([test_pred_base[k] for k in SPEC])

train_pred = np.clip(meta.predict(Ttr), 0.0, 5.0)
test_pred  = np.clip(meta.predict(Tte), 0.0, 5.0)

TRAIN_RMSE = rmse(y, train_pred)
TRAIN_PEARSON = pearsonr(y, train_pred)[0]
CV_RMSE = rmse(y, stack_oof)
CV_PEARSON = pearsonr(y, stack_oof)[0]
BASELINE_RMSE = np.sqrt(np.mean((y - y.mean())**2))

print(f"Full Training RMSE (fit on train) : {TRAIN_RMSE:.4f}")
print(f"Full Training Pearson Correlation : {TRAIN_PEARSON:.4f}")
print(f"Cross-Validated RMSE (honest)     : {CV_RMSE:.4f}")
print(f"Cross-Validated Pearson (honest)  : {CV_PEARSON:.4f}")
print(f"Baseline RMSE (mean predictor)    : {BASELINE_RMSE:.4f}")
print("="*65)

# ==========================================
# 8. GENERATE SUBMISSION FILE
# ==========================================
sub_out = pd.DataFrame({
    "filename": test_df[FILE_COL],
    "label": test_pred
})

submission_path = r"e:\SHL\submission.csv"
sub_out.to_csv(submission_path, index=False)

print(f"\nSaved final submission to: {submission_path}")
print(f"Submission Shape   : {sub_out.shape}")
print(f"Prediction Range   : [{sub_out['label'].min():.3f}, {sub_out['label'].max():.3f}]")
print(f"Prediction Mean    : {sub_out['label'].mean():.3f}")
print(f"Prediction Std     : {sub_out['label'].std():.3f}")
print("\nFirst 10 submission rows:")
print(sub_out.head(10))
print("\nPIPELINE COMPLETED SUCCESSFULLY!")
