import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ---------------- #
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR = "data/test"
MODEL_PATH = "models/plant_model.h5"   # or .keras if you change format

# ---------------- LOAD MODEL ---------------- #
print("📦 Loading model...")
model = load_model(MODEL_PATH)

# ---------------- LOAD TEST DATA ---------------- #
print("📂 Loading test dataset...")
test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ---------------- EVALUATE ---------------- #
print("\n🧪 Evaluating model...\n")
test_loss, test_acc = model.evaluate(test_ds)

print("\n==============================")
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"❌ Test Loss: {test_loss:.4f}")
print("==============================\n")

# ---------------- PREDICTIONS ---------------- #
y_true = []
y_pred = []

print("🔍 Running predictions...")
for images, labels in test_ds:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

class_names = test_ds.class_names

# ---------------- REPORT ---------------- #
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ---------------- CONFUSION MATRIX ---------------- #
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(16,14))
sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

