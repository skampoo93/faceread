"""
Genere les visualisations : matrice de confusion, courbes d'entrainement, T-SNE.
A lancer juste apres train_advanced.py (utilise model et history en memoire).
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ── 1. PRÉDICTIONS NORMALES ────────────────────────────────────────
y_pred     = model.predict(X_test, verbose=0)
y_pred_cls = np.argmax(y_pred, axis=1)
y_true_cls = np.argmax(y_test, axis=1)

# ── 2. PRÉDICTIONS TTA ────────────────────────────────────────────
import tensorflow as tf
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
tta_preds    /= 10
y_pred_tta    = np.argmax(tta_preds, axis=1)

# ── 3. MATRICE DE CONFUSION ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, preds, title in zip(axes,
    [y_pred_cls, y_pred_tta],
    ['Matrice de confusion — Normal', 'Matrice de confusion — TTA']):
    cm = confusion_matrix(y_true_cls, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Réel')
    ax.set_xlabel('Prédit')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("Matrice sauvegardée !")

# ── 4. COURBES LOSS / ACCURACY ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='Train', color='#7F77DD')
axes[0].plot(history.history['val_accuracy'], label='Val',   color='#1D9E75')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', color='#7F77DD')
axes[1].plot(history.history['val_loss'], label='Val',   color='#1D9E75')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Courbes d\'entraînement — CBAM + Mixup', fontsize=14)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print("Courbes sauvegardées !")

# ── 5. RAPPORT PAR ÉMOTION ────────────────────────────────────────
print("\nRapport détaillé (TTA) :")
print(classification_report(y_true_cls, y_pred_tta, target_names=EMOTIONS))

# ── 6. T-SNE ─────────────────────────────────────────────────────
print("\nCalcul T-SNE (peut prendre 2-3 minutes)...")
feature_extractor = Model(
    inputs=model.input,
    outputs=model.layers[-3].output  # avant la dernière Dense
)
features = feature_extractor.predict(X_test[:2000], verbose=0)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(features)

colors = ['#E24B4A','#8B4513','#7F77DD','#1D9E75','#888888','#4A90D9','#EF9F27']
plt.figure(figsize=(12, 8))
for i, (emotion, color) in enumerate(zip(EMOTIONS, colors)):
    mask = y_true_cls[:2000] == i
    plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                c=color, label=emotion, alpha=0.6, s=15)
plt.title('T-SNE — Représentations apprises par le CNN', fontsize=14)
plt.legend(markerscale=2)
plt.tight_layout()
plt.savefig('tsne.png', dpi=150)
plt.show()
print("T-SNE sauvegardé !")
