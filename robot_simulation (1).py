
import pygame
import numpy as np
import librosa
import sounddevice as sd
import joblib
import random

# ======================
# LOAD MODEL
# ======================

model = joblib.load("voice_model.pkl")
scaler = joblib.load("scaler.pkl")

# ======================
# CONFIG
# ======================

CLASSES = [
    "tien",
    "trai",
    "phai",
    "dung"
]

MIC_ID = 1

SR = 16000

TIME = 1.2

CONF = 0.50

# ======================
# PYGAME
# ======================

pygame.init()

WIDTH = 900
HEIGHT = 600

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "Voice Car Game"
)

clock = pygame.time.Clock()

font = pygame.font.SysFont(
    "Arial",
    30
)

# ======================
# XE
# ======================

x = WIDTH // 2
y = HEIGHT // 2

angle = 0

speed = 0

MOVE_SPEED = 3

score = 0

last_cmd = "None"

# ======================
# FOOD
# ======================

foods = []

for i in range(15):

    foods.append([

        random.randint(50, WIDTH - 50),

        random.randint(50, HEIGHT - 50)

    ])

# ======================
# FEATURE
# ======================

def feature(audio):

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

    for m in [
        mfcc,
        delta,
        delta2
    ]:

        feat.extend(
            np.mean(m, axis=1)
        )

        feat.extend(
            np.std(m, axis=1)
        )

    return np.array(feat)

# ======================
# PREPARE
# ======================

def prepare(audio):

    audio, _ = librosa.effects.trim(
        audio,
        top_db=20
    )

    if len(audio) < 1000:

        return None

    if len(audio) < SR:

        audio = np.pad(
            audio,
            (0, SR - len(audio))
        )

    else:

        audio = audio[:SR]

    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:

        audio = audio / peak

    return audio

# ======================
# LISTEN
# ======================

def listen():

    print("🎙️ Đang nghe...")

    audio = sd.rec(
        int(TIME * SR),
        samplerate=SR,
        channels=1,
        dtype="float32",
        device=MIC_ID
    )

    sd.wait()

    audio = audio.flatten()

    rms = np.sqrt(
        np.mean(audio**2)
    )

    print(
        "RMS =",
        round(rms, 4)
    )

    if rms < 0.003:

        print("🔇 Im lặng")

        return None

    audio = prepare(audio)

    if audio is None:

        return None

    feat = feature(audio)

    feat = feat.reshape(1, -1)

    feat = scaler.transform(feat)

    proba = model.predict_proba(
        feat
    )[0]

    idx = np.argmax(proba)

    conf = proba[idx]

    label = CLASSES[idx]

    print(
        f"{label} "
        f"{conf*100:.1f}%"
    )

    if conf < CONF:

        print("❌ Không chắc")

        return None

    return label

# ======================
# MAIN
# ======================

running = True

while running:

    # ======================
    # EVENT
    # ======================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

        # ENTER để nói
        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_RETURN:

                label = listen()

                print(
                    "Kết quả:",
                    label
                )

                if label is not None:

                    last_cmd = label

                # ======================
                # COMMAND
                # ======================

                if label == "tien":

                    speed = MOVE_SPEED

                    print("🚗 CHẠY")

                elif label == "dung":

                    speed = 0

                    print("🛑 DỪNG")

                elif label == "trai":

                    angle += 45

                    print("⬅️ TRÁI")

                elif label == "phai":

                    angle -= 45

                    print("➡️ PHẢI")

    # ======================
    # MOVE
    # ======================

    rad = np.radians(angle)

    x += speed * np.cos(rad)

    y -= speed * np.sin(rad)

    # ======================
    # ĐÂM TƯỜNG
    # ======================

    if (
        x < 20 or
        x > WIDTH - 20 or
        y < 20 or
        y > HEIGHT - 20
    ):

        x = WIDTH // 2

        y = HEIGHT // 2

        speed = 0

        print("💥 Đâm tường")

    # ======================
    # ĂN ĐIỂM
    # ======================

    for food in foods[:]:

        fx, fy = food

        dist = np.sqrt(
            (x - fx)**2 +
            (y - fy)**2
        )

        if dist < 25:

            foods.remove(food)

            foods.append([

                random.randint(
                    50,
                    WIDTH - 50
                ),

                random.randint(
                    50,
                    HEIGHT - 50
                )

            ])

            score += 1

    # ======================
    # DRAW
    # ======================

    screen.fill((220, 220, 220))

    # FOOD
    for food in foods:

        pygame.draw.circle(
            screen,
            (255, 0, 0),
            food,
            6
        )

    # ======================
    # XE
    # ======================

    car_surface = pygame.Surface(
        (50, 25),
        pygame.SRCALPHA
    )

    # THÂN XE
    pygame.draw.rect(
        car_surface,
        (0, 0, 255),
        (0, 3, 50, 20),
        border_radius=5
    )

    # KÍNH
    pygame.draw.rect(
        car_surface,
        (150, 220, 255),
        (12, 6, 25, 12),
        border_radius=3
    )

    # BÁNH
    pygame.draw.circle(
        car_surface,
        (0, 0, 0),
        (10, 3),
        4
    )

    pygame.draw.circle(
        car_surface,
        (0, 0, 0),
        (40, 3),
        4
    )

    pygame.draw.circle(
        car_surface,
        (0, 0, 0),
        (10, 22),
        4
    )

    pygame.draw.circle(
        car_surface,
        (0, 0, 0),
        (40, 22),
        4
    )

    rotated = pygame.transform.rotate(
        car_surface,
        angle
    )

    rect = rotated.get_rect(
        center=(x, y)
    )

    screen.blit(
        rotated,
        rect
    )

    # ======================
    # TEXT
    # ======================

    text1 = font.render(
        f"Command: {last_cmd}",
        True,
        (0, 0, 0)
    )

    screen.blit(
        text1,
        (20, 20)
    )

    text2 = font.render(
        f"Score: {score}",
        True,
        (0, 0, 0)
    )

    screen.blit(
        text2,
        (20, 60)
    )

    pygame.display.update()

    clock.tick(60)

pygame.quit()

