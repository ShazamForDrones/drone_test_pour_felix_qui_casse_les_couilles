import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.feature
import librosa.display

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, Callback
from keras.utils import to_categorical

from google.colab import drive

# =================== MOUNT GOOGLE DRIVE ===================
drive.mount('/content/drive')

# =================== CONSTANTES ===================
# Pour 10k fichiers, on réduit un peu la taille pour tenir en RAM
SR = 22050          # sample rate
DURATION = 3.0      # 3 secondes par extrait (au lieu de 10)
N_MELS = 64         # moins de mels pour réduire la taille
HOP_LENGTH = 1024   # frames plus espacées => moins de colonnes
MAX_LEN = 200       # longueur temporelle max après mels (ajuster au besoin)



# =================== SEEDS ===================
np.random.seed(42)
tf.random.set_seed(42)

# =================== GPU CHECK ===================
print("\n===== TensorFlow GPU DIAGNOSTIC =====")
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')
print("CPUs available:", cpus)
print("GPUs available:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU detected and memory growth enabled.")
    except Exception as e:
        print("⚠️ Could not set memory growth:", e)
else:
    print("❌ No GPU detected by TensorFlow. Training will run on CPU only.")
print("======================================\n")

# =================== CLASSES ===================
things = ['unknown', 'yes_drone']  # ordre des classes

# =================== CALLBACK D'AFFICHAGE ===================
class TrainingPrinter(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n🔵 START Epoch {epoch + 1}")
        self.batch_losses = []
        self.batch_acc = []

    def on_batch_end(self, batch, logs=None):
        # On n'affiche pas tous les batchs pour éviter de ralentir
        if batch % 10 != 0:
            return
        loss = logs.get('loss')
        acc = logs.get('accuracy')
        self.batch_losses.append(loss)
        self.batch_acc.append(acc)
        print(f"  Batch {batch:04d} → loss={loss:.4f}  acc={acc:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        if self.batch_losses:
            avg_loss = np.mean(self.batch_losses)
            avg_acc = np.mean(self.batch_acc)
        else:
            avg_loss = logs.get('loss')
            avg_acc = logs.get('accuracy')

        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')

        print(f"\n🟢 END Epoch {epoch + 1}")
        print(f"   • Train avg loss: {avg_loss:.4f}, avg acc: {avg_acc:.4f}")
        print(f"   • Val   loss:     {val_loss:.4f}, val acc: {val_acc:.4f}")

# =================== DATA AUGMENTATION ===================
def augment_audio(mel_db):
    """
    Version plus légère : original + bruit léger seulement
    => x2 au lieu de x4, pour limiter la RAM avec 10k fichiers.
    """
    augmented = [mel_db]  # Original

    # Bruit léger
    noise = np.random.normal(0, 0.005, mel_db.shape)
    augmented.append(mel_db + noise)

    return augmented  # 2 versions (original + 1 augmentée)

# =================== CHARGEMENT DES DONNÉES ===================
X_raw, y_raw = [], []

print("Chargement des données...")

folder = "/content/drive/MyDrive/dataset/Binary_Drone_Audio"
if not os.path.exists(folder):
    print(f"⚠️  ATTENTION: Dossier '{folder}' introuvable!")
else:
    subfolders = [
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d))
    ]
    print(f"Sous-dossiers trouvés dans '{folder}': {subfolders}")

    for class_name in subfolders:
        class_dir = os.path.join(folder, class_name)

        if class_name not in things:
            print(f"⚠️  Sous-dossier ignoré (classe inconnue) : {class_name}")
            continue

        label = things.index(class_name)

        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.wav', '.mp3', '.flac'))
        ]

        print(f"Classe '{class_name}' → {len(files)} fichiers")

        for filename in files:
            song_path = os.path.join(class_dir, filename)

            try:
                # Charge DURATION secondes en mono à SR Hz
                y_audio, sr = librosa.load(song_path, duration=DURATION, sr=SR, mono=True)

                # Melspectrogramme
                mel = librosa.feature.melspectrogram(
                    y=y_audio,
                    sr=sr,
                    n_mels=N_MELS,
                    n_fft=2048,
                    hop_length=HOP_LENGTH
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)

                # Padding / tronquage sur l'axe temps
                if mel_db.shape[1] < MAX_LEN:
                    pad_width = MAX_LEN - mel_db.shape[1]
                    mel_db = np.pad(
                        mel_db,
                        pad_width=((0, 0), (0, pad_width)),
                        mode='constant'
                    )
                else:
                    mel_db = mel_db[:, :MAX_LEN]

                # Stocker seulement les originaux pour split avant augmentation
                X_raw.append(mel_db)
                y_raw.append(label)

            except Exception as e:
                print(f"Erreur avec {song_path}: {e}")
                continue

    print(f"✅ Chargé {len(X_raw)} fichiers originaux avant augmentation")

# Petit résumé dataset
y_raw = np.array(y_raw)
for idx, name in enumerate(things):
    count = np.sum(y_raw == idx)
    print(f"Classe {name} ({idx}) : {count} fichiers")

# =================== SPLIT TRAIN / TEST AVANT AUGMENTATION ===================
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw,
    y_raw,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

print(f"Split: {len(X_train_raw)} train originaux, {len(X_test_raw)} test originaux")

# =================== AUGMENTATION SUR LE TRAIN UNIQUEMENT ===================
X_train, y_train = [], []
for mel_db, label in zip(X_train_raw, y_train_raw):
    augmented_samples = augment_audio(mel_db)
    for aug_mel in augmented_samples:
        X_train.append(aug_mel)
        y_train.append(label)

X_test = np.array(X_test_raw, dtype=np.float32)
y_test = np.array(y_test_raw, dtype=np.int32)

print(f"✅ Après augmentation: {len(X_train)} train samples, {len(X_test)} test samples")

# =================== NORMALISATION & RESHAPE ===================
X_train = np.array(X_train, dtype=np.float32)

# Normalisation basée sur le train uniquement
train_min = X_train.min()
train_max = X_train.max()
denom = (train_max - train_min + 1e-8)

X_train = (X_train - train_min) / denom
X_test = (X_test - train_min) / denom

# Ajout du channel (1) pour Conv2D
X_train = X_train[..., np.newaxis]  # (N, N_MELS, MAX_LEN, 1)
X_test = X_test[..., np.newaxis]

# One-hot sur les labels
y_train = to_categorical(np.array(y_train), num_classes=len(things))
y_test = to_categorical(np.array(y_test), num_classes=len(things))

print("Shapes :")
print("X_train :", X_train.shape)
print("y_train :", y_train.shape)
print("X_test  :", X_test.shape)
print("y_test  :", y_test.shape)

# =================== MODELE CNN ===================
model = Sequential([
    Input(shape=(N_MELS, MAX_LEN, 1)),

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

model.summary()

# =================== CALLBACKS ===================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,              # un peu moins de patience
    restore_best_weights=True,
    verbose=1
)

printer = TrainingPrinter()

# =================== TRAINING ===================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,               # on limite à 30, early stopping arrêtera plus tôt
    batch_size=32,
    callbacks=[early_stop, printer],
    verbose=0   # c'est le callback qui gère l'affichage
)

# =================== EVALUATION ===================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("\n" + "=" * 60)
print(f"TEST ACCURACY: {test_acc * 100:.2f}%")
print("=" * 60)

# =================== SAVE MODEL ===================
timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
model_name = f'binary_drone_classifier_{timestamp}.keras'
model.save(model_name)
print(f"✅ Modèle sauvegardé sous : {model_name}")

# =================== COURBES D'APPRENTISSAGE ===================
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()
plt.grid(True)

# Loss
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
