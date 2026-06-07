import sounddevice as sd
import numpy as np
import librosa
import os
import scipy.io.wavfile as wav

DEVICE_ID   = 1      # đổi sang mic Realtek nếu cần
SAMPLE_RATE = 48000
TARGET_SR   = 16000
TARGET_LEN  = 16000
DURATION    = 1

COMMANDS = ["trai","phai"]

LABELS = {
    "trai": "TRÁI",
    "phai": "PHẢI"
   
}

SAMPLES = 30


def prepare(audio):

    audio = librosa.resample(
        audio,
        orig_sr=SAMPLE_RATE,
        target_sr=TARGET_SR
    )

    audio, _ = librosa.effects.trim(
        audio,
        top_db=20
    )

    if len(audio) < TARGET_LEN:
        audio = np.pad(
            audio,
            (0, TARGET_LEN - len(audio))
        )
    else:
        audio = audio[:TARGET_LEN]

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak * 0.9

    return (audio * 32767).astype(np.int16)


os.makedirs("my_dataset", exist_ok=True)

for cmd in COMMANDS:
    os.makedirs(
        f"my_dataset/{cmd}",
        exist_ok=True
    )

print(" THU DATASET LỆNH")
print()

for cmd in COMMANDS:

    folder = f"my_dataset/{cmd}"

    existing = len([
        f for f in os.listdir(folder)
        if f.endswith(".wav")
    ])

    print("=" * 50)
    print(f" Lệnh: {LABELS[cmd]}")
    print(f" Đã có: {existing} file")
    print(f" Thu thêm: {SAMPLES} file")
    print()
    print("d + Enter = xóa file vừa thu")
    print()

    input(" Nhấn Enter để bắt đầu...")

    count = existing
    last_path = None

    while count < existing + SAMPLES:

        inp = input(
            f"\n[{count+1}/{existing+SAMPLES}] Enter để thu (d=xóa): "
        ).strip().lower()

        if inp == "d":

            if last_path and os.path.exists(last_path):

                os.remove(last_path)

                count -= 1

                print(
                    f"🗑️ Đã xóa {os.path.basename(last_path)}"
                )

                last_path = None

            else:
                print(" Không có file để xóa")

            continue

        print(
            f" Đang ghi âm {DURATION}s..."
        )

        audio = sd.rec(
            int(SAMPLE_RATE * DURATION),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=DEVICE_ID
        )

        sd.wait()

        audio = audio.flatten()

        rms = np.sqrt(
            np.mean(audio ** 2)
        )

        print(f"   RMS = {rms:.5f}")

        if rms < 0.01:
            print(" Tiếng quá nhỏ, thu lại")
            continue

        audio_ready = prepare(audio)

        filename = f"{cmd}_{count:04d}.wav"

        path = os.path.join(
            folder,
            filename
        )

        wav.write(
            path,
            TARGET_SR,
            audio_ready
        )

        last_path = path

        count += 1

        print("Đã lưu")

    print()
    print(f"🎉 Hoàn thành lớp {cmd}")

print()
print(" Đã thu xong")
print(" Chạy train.py để train lại model")
