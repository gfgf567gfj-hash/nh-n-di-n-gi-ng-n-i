import numpy as np
import librosa
import sounddevice as sd
import joblib

# =====================
# CONFIG
# =====================

MODEL = "voice_model.pkl"
SCALER = "scaler.pkl"

CLASSES = ["tien", "trai", "phai", "dung"]

DISPLAY = {
    "tien": "⬆️ TIẾN",
    "trai": "⬅️ TRÁI",
    "phai": "➡️ PHẢI",
    "dung": "⏹️ DỪNG"
}

MIC_ID = 1
RECORD_TIME = 1.2

TARGET_SR = 16000
TARGET_LEN = 16000

# =====================
# LOAD
# =====================

model = joblib.load(MODEL)
scaler = joblib.load(SCALER)

print("✅ Model loaded")
print("Feature count:", scaler.n_features_in_)

# =====================
# PREPROCESS
# =====================

def prepare(audio, sr):

    if sr != TARGET_SR:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=TARGET_SR
        )

    audio, _ = librosa.effects.trim(
        audio,
        top_db=20
    )

    if len(audio) < TARGET_LEN:
        audio = np.pad(
            audio,
            (0, TARGET_LEN - len(audio)))
    else:
        audio = audio[:TARGET_LEN]

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    return audio

# =====================
# FEATURE
# =====================

def extract_feature(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=20
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = []

    for m in [mfcc, delta, delta2]:
        feat.extend(np.mean(m, axis=1))
        feat.extend(np.std(m, axis=1))

    return np.array(feat)

# =====================
# MIC
# =====================

info = sd.query_devices(MIC_ID)
MIC_SR = int(info["default_samplerate"])

print("🎙️", info["name"])
print("Sample Rate:", MIC_SR)

# =====================
# LOOP
# =====================

while True:

    cmd = input(
        "\nEnter để ghi âm | q để thoát: "
    ).strip().lower()

    if cmd == "q":
        break

    print("🔴 Đang ghi âm...")

    audio = sd.rec(
        int(RECORD_TIME * MIC_SR),
        samplerate=MIC_SR,
        channels=1,
        dtype="float32",
        device=MIC_ID
    )

    sd.wait()

    audio = audio.flatten()

    rms = np.sqrt(np.mean(audio**2))

    print("RMS =", round(rms, 4))

    if rms < 0.005:
        print("🔇 Không phát hiện giọng nói")
        continue

    audio = prepare(audio, MIC_SR)

    feat = extract_feature(audio).reshape(1, -1)

    print("Feature:", feat.shape)

    feat = scaler.transform(feat)

    proba = model.predict_proba(feat)[0]

    idx = np.argmax(proba)

    label = CLASSES[idx]
    conf = proba[idx]

    print("\n======================")
    print(DISPLAY[label])
    print(f"Độ tin cậy: {conf*100:.1f}%")

    print("\nChi tiết:")

    for c, p in zip(CLASSES, proba):
        print(f"{c:5s}: {p*100:6.2f}%")

    print("======================")