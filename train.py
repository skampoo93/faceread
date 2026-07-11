"""
Entrainement du modele de base (CNN v2 dans le README).
Necessite FER2013 deja telecharge dans data/train et data/test
(voir prepare_data.py).
"""
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ── 1. DONNÉES ─────────────────────────────────────────────────────
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE  = 48

def load_images(folder):
    X, y = [], []
    for label, emotion in enumerate(EMOTIONS):
        path = os.path.join(folder, emotion)
        if not os.path.exists(path):
            continue
        for fname in os.listdir(path):
            img = cv2.imread(os.path.join(path, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img.reshape(IMG_SIZE, IMG_SIZE, 1))
            y.append(label)
    return np.array(X), np.array(y)

print("Chargement...")
X_all, y_all       = load_images('data/train')
X_test, y_test_raw = load_images('data/test')
X_all  = X_all  / 255.0
X_test = X_test / 255.0
y_all_cat = to_categorical(y_all,      num_classes=7)
y_test    = to_categorical(y_test_raw, num_classes=7)
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all_cat, test_size=0.15, random_state=42
)
print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# ── 2. ARCHITECTURE ────────────────────────────────────────────────
def build_model():
    inputs = keras.Input(shape=(48, 48, 1))

    x = layers.Conv2D(64, (3,3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(512, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(7, activation='softmax')(x)

    return keras.Model(inputs, outputs)

model = build_model()

# ── 3. AUGMENTATION ────────────────────────────────────────────────
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(X_train)

# ── 4. COMPILATION — learning rate réduit ──────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── 5. CALLBACKS ───────────────────────────────────────────────────
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_model_v2.keras',
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
]

# ── 6. ENTRAÎNEMENT ────────────────────────────────────────────────
print("\nEntraînement en cours...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# ── 7. ÉVALUATION ──────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy  : {acc:.2%}")
print(f"Test loss      : {loss:.4f}")
print(f"Meilleure val  : {max(history.history['val_accuracy']):.2%}")
