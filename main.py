import librosa.feature
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import datetime

np.random.seed(42)
tf.random.set_seed(42)

things = ['background', 'drones', 'helicopter']


def augment_audio(mel_db):
    """Augmente TOUJOURS - pas de random!"""
    augmented = [mel_db]  # Original

    # Bruit léger (TOUJOURS)
    noise = np.random.normal(0, 0.005, mel_db.shape)
    augmented.append(mel_db + noise)

    # Time shift (TOUJOURS)
    shift = np.random.randint(-40, 40)
    augmented.append(np.roll(mel_db, shift, axis=1))

    # Pitch shift (TOUJOURS)
    pitch = np.random.uniform(0.95, 1.05)
    augmented.append(mel_db * pitch)

    return augmented  # TOUJOURS 4 versions (1 original + 3 augmentées)


X_raw, y_raw = [], []

print("Chargement des données...")

# Boucle sur tous les fichiers dans le dossier Audio
folder = "Audio"
if not os.path.exists(folder):
    print(f"⚠️  ATTENTION: Dossier '{folder}' introuvable!")
else:
    files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.flac'))]
    print(f"Trouvé {len(files)} fichiers audio dans '{folder}'")

    for filename in files:
        song_path = os.path.join(folder, filename)

        # Déterminer la classe basée sur le nom du fichier
        label = None
        filename_lower = filename.lower()

        if 'background' in filename_lower:
            label = things.index('background')
        elif 'drone' in filename_lower or 'drones' in filename_lower:
            label = things.index('drones')
        elif 'helicopter' in filename_lower or 'heli' in filename_lower:
            label = things.index('helicopter')
        else:
            print(f"⚠️  Impossible de déterminer la classe pour: {filename}")
            continue

        try:
            y_audio, sr = librosa.load(song_path, duration=10, sr=22050)

            # Création spectrogramme
            mel = librosa.feature.melspectrogram(
                y=y_audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Padding/truncation
            max_len = 660
            if mel_db.shape[1] < max_len:
                pad_width = max_len - mel_db.shape[1]
                mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_db = mel_db[:, :max_len]

            # Stocker seulement les originaux pour split avant augmentation
            X_raw.append(mel_db)
            y_raw.append(label)

        except Exception as e:
            print(f"Erreur avec {filename}: {e}")
            continue

    print(f"✅ Chargé {len(X_raw)} fichiers originaux avant augmentation")


# 🔹 SPLIT AVANT AUGMENTATION 🔹
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, shuffle=True, random_state=42
)

# 🔹 AUGMENTATION UNIQUEMENT SUR LE TRAIN 🔹
X_train, y_train = [], []
for mel_db, label in zip(X_train_raw, y_train_raw):
    augmented_samples = augment_audio(mel_db)
    for aug_mel in augmented_samples:
        X_train.append(aug_mel)
        y_train.append(label)

# 🔹 TEST SET SANS AUGMENTATION 🔹
X_test = np.array(X_test_raw)
y_test = np.array(y_test_raw)

print(f"✅ Après augmentation: {len(X_train)} train samples, {len(X_test)} test samples")

# Normalisation et reshape
X_train = np.array(X_train)
X_test = np.array(X_test)

# Normaliser entre 0 et 1
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# One-hot labels
y_train = to_categorical(np.array(y_train))
y_test = to_categorical(np.array(y_test))


# =================== MODEL ===================
model = Sequential([
    Input(shape=(128, 660, 1)),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.1),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.1),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(len(things), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\n{'='*60}")
print(f"TEST ACCURACY: {test_acc*100:.2f}%")
print(f"{'='*60}")

model.save(f'genre_classifier_{datetime.datetime.now().strftime("%m-%d_%H-%M-%S")}.keras')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
