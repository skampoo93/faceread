"""
Entrainement avance : CBAM + Mixup + TTA.
Cette version a produit best_model_v4.keras et le score de 67.30% (avec TTA).
Necessite FER2013 deja telecharge dans data/train et data/test.
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

# ── 2. CBAM ────────────────────────────────────────────────────────
def cbam_block(x):
    filters = x.shape[-1]
    avg = layers.GlobalAveragePooling2D()(x)
    mx  = layers.GlobalMaxPooling2D()(x)
    avg = layers.Dense(filters // 8, activation='relu')(avg)
    avg = layers.Dense(filters, activation='sigmoid')(avg)
    mx  = layers.Dense(filters // 8, activation='relu')(mx)
    mx  = layers.Dense(filters, activation='sigmoid')(mx)
    ch  = layers.Add()([avg, mx])
    ch  = layers.Reshape((1, 1, filters))(ch)
    x   = layers.Multiply()([x, ch])
    avg_s = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_s = layers.Lambda(lambda t: tf.reduce_max(t,  axis=-1, keepdims=True))(x)
    sp    = layers.Concatenate()([avg_s, max_s])
    sp    = layers.Conv2D(1, (7,7), padding='same', activation='sigmoid')(sp)
    x     = layers.Multiply()([x, sp])
    return x

# ── 3. RESIDUAL BLOCK + CBAM ───────────────────────────────────────
def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = cbam_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

# ── 4. ARCHITECTURE ────────────────────────────────────────────────
def build_model():
    inputs = keras.Input(shape=(48, 48, 1))
    x = layers.Conv2D(64, (3,3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = layers.MaxPooling2D(2,2)(x)
    x = layers.Dropout(0.25)(x)
    x = residual_block(x, 512)
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

# ── 5. MIXUP ───────────────────────────────────────────────────────
def mixup_generator(X, y, batch_size=64, alpha=0.2):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    gen = datagen.flow(X, y, batch_size=batch_size)
    while True:
        X1, y1 = next(gen)
        X2, y2 = next(gen)
        l = min(len(X1), len(X2))
        X1, y1, X2, y2 = X1[:l], y1[:l], X2[:l], y2[:l]
        lam = np.random.beta(alpha, alpha)
        yield lam * X1 + (1 - lam) * X2, lam * y1 + (1 - lam) * y2

# ── 6. COMPILATION ─────────────────────────────────────────────────
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── 7. CALLBACKS — ReduceLROnPlateau stable ────────────────────────
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_model_v4.keras',
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
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
]

# ── 8. ENTRAÎNEMENT ────────────────────────────────────────────────
print("\nEntraînement CBAM + Mixup...")
steps = len(X_train) // 64
history = model.fit(
    mixup_generator(X_train, y_train, batch_size=64),
    steps_per_epoch=steps,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# ── 9. TTA ─────────────────────────────────────────────────────────
print("\nTTA en cours...")
tta_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    horizontal_flip=True
)
tta_preds = np.zeros((len(X_test), 7))
for _ in range(10):
    tta_preds += model.predict(
        tta_datagen.flow(X_test, batch_size=64, shuffle=False), verbose=0
    )
tta_preds /= 10
tta_acc = np.mean(np.argmax(tta_preds, axis=1) == np.argmax(y_test, axis=1))

# ── 10. RÉSULTATS ──────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy (normal) : {acc:.2%}")
print(f"Test accuracy (TTA)    : {tta_acc:.2%}")
print(f"Meilleure val          : {max(history.history['val_accuracy']):.2%}")

# ── 11. SAUVEGARDE DE L'HISTORIQUE ─────────────────────────────────
import json
with open('history.json', 'w') as f:
    json.dump(history.history, f)
print("Historique sauvegardé dans history.json")
