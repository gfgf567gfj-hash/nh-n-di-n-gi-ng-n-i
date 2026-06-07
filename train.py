import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ======================
# CONFIG
# ======================

DATASET = "my_dataset"

CLASSES = [
    "tien",
    "trai",
    "phai",
    "dung"
]

SR = 16000
LENGTH = 16000

# ======================
# PREPROCESS
# ======================

def prepare_audio(audio, sr):

    if sr != SR:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SR
        )

    audio, _ = librosa.effects.trim(
        audio,
        top_db=20
    )

    if len(audio) < LENGTH:
        audio = np.pad(
            audio,
            (0, LENGTH - len(audio))
        )
    else:
        audio = audio[:LENGTH]

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    return audio

# ======================
# FEATURE EXTRACTION
# ======================

def extract_feature(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=20
    )

    delta = librosa.feature.delta(mfcc)

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    feat = []

    for m in [mfcc, delta, delta2]:

        feat.extend(
            np.mean(m, axis=1)
        )

        feat.extend(
            np.std(m, axis=1)
        )

    return np.array(feat)

# ======================
# LOAD DATASET
# ======================

X = []
y = []

print("Loading dataset...")

for label, cls in enumerate(CLASSES):

    folder = os.path.join(DATASET, cls)

    if not os.path.exists(folder):
        print(f"Không tìm thấy thư mục: {folder}")
        continue

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".wav")
    ]

    print(f"{cls}: {len(files)} files")

    for file in files:

        path = os.path.join(
            folder,
            file
        )

        try:

            audio, sr = librosa.load(
                path,
                sr=None
            )

            audio = prepare_audio(
                audio,
                sr
            )

            X.append(
                extract_feature(audio)
            )

            y.append(label)

            # Noise augmentation

            noise = np.random.normal(
                0,
                0.003,
                len(audio)
            )

            audio_aug = audio + noise

            X.append(
                extract_feature(audio_aug)
            )

            y.append(label)

        except Exception as e:
            print("Lỗi:", path)
            print(e)

X = np.array(X)
y = np.array(y)

print("\nDataset shape:", X.shape)

# ======================
# TRAIN TEST SPLIT
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================
# SCALE
# ======================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

# ======================
# TRAIN
# ======================

print("\nTraining SVM...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True
)

model.fit(
    X_train,
    y_train
)

# ======================
# EVALUATE
# ======================

pred = model.predict(
    X_test
)

print("\nAccuracy:",
      accuracy_score(y_test, pred))

print()

print(
    classification_report(
        y_test,
        pred,
        target_names=CLASSES
    )
)

# ======================
# SAVE
# ======================

joblib.dump(
    model,
    "voice_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("✅ Đã lưu voice_model.pkl")
print("✅ Đã lưu scaler.pkl")
print("Feature count:", scaler.n_features_in_)