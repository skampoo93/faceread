import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import os

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE  = 48

def load_images(folder):
    X, y = [], []
    for label, emotion in enumerate(EMOTIONS):
        path = os.path.join(folder, emotion)
        if not os.path.exists(path):
            print(f"Dossier manquant : {path}")
            continue
        files = os.listdir(path)
        print(f"{emotion} : {len(files)} images")
        for fname in files:
            img_path = os.path.join(path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img.reshape(IMG_SIZE, IMG_SIZE, 1))
            y.append(label)
    return np.array(X), np.array(y)

print("Chargement train...")
X_train_all, y_train_all = load_images('data/train')

print("\nChargement test...")
X_test, y_test = load_images('data/test')

X_train_all = X_train_all / 255.0
X_test      = X_test      / 255.0

y_train_all = to_categorical(y_train_all, num_classes=7)
y_test      = to_categorical(y_test,      num_classes=7)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all, test_size=0.15, random_state=42
)

print(f"\nTrain : {X_train.shape[0]} images")
print(f"Val   : {X_val.shape[0]} images")
print(f"Test  : {X_test.shape[0]} images")

os.makedirs('data/npy', exist_ok=True)
np.save('data/npy/X_train.npy', X_train)
np.save('data/npy/X_val.npy',   X_val)
np.save('data/npy/X_test.npy',  X_test)
np.save('data/npy/y_train.npy', y_train)
np.save('data/npy/y_val.npy',   y_val)
np.save('data/npy/y_test.npy',  y_test)
print("Donnees sauvegardees dans data/npy/")

counts = [len(os.listdir(f'data/train/{e}')) for e in EMOTIONS]
plt.figure(figsize=(10, 4))
plt.bar(EMOTIONS, counts, color='#7F77DD')
plt.title('Distribution FER2013')
plt.ylabel("Nombre d'images")
plt.tight_layout()
plt.savefig('data/distribution.png')
print("Graphique sauvegarde !")
