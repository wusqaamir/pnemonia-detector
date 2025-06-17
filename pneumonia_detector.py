import subprocess
import sys

packages = ['opencv-python', 'scikit-learn', 'matplotlib']

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
import os
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Step 1: Create dummy image folders ---
os.makedirs('dummy_data/NORMAL', exist_ok=True)
os.makedirs('dummy_data/PNEUMONIA', exist_ok=True)

# --- Step 2: Generate fake images ---
def create_dummy_image(save_path):
    img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)  # grayscale
    cv2.imwrite(save_path, img)

# Create 10 fake NORMAL images
for i in range(10):
    create_dummy_image(f'dummy_data/NORMAL/normal_{i}.jpg')

# Create 10 fake PNEUMONIA images
for i in range(10):
    create_dummy_image(f'dummy_data/PNEUMONIA/pneumonia_{i}.jpg')

print("✅ Dummy images created!")

# --- Step 3: Load and label the dummy images ---
def load_images(folder, label):
    data = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            data.append(img.flatten())  # convert to 1D feature vector
            labels.append(label)
    return data, labels

normal_data, normal_labels = load_images('dummy_data/NORMAL', 0)
pneumonia_data, pneumonia_labels = load_images('dummy_data/PNEUMONIA', 1)

# Combine
X = np.array(normal_data + pneumonia_data)
y = np.array(normal_labels + pneumonia_labels)

# --- Step 4: Train and test your AI model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Dummy model trained! Accuracy: {acc * 100:.2f}%")
# --- Predict on one image ---
def predict_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found.")
        return
    img = cv2.resize(img, (100, 100)).flatten().reshape(1, -1)
    prediction = model.predict(img)[0]
    label = "PNEUMONIA" if prediction == 1 else "NORMAL"
    print(f"🧠 Prediction: This X-ray is likely: {label}")

# Run prediction
predict_image("dummy_data/NORMAL/normal_0.jpg")
import joblib
joblib.dump(model, "pneumonia_model.pkl")

