from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ---------------- CONFIG ---------------- #
MODEL_PATH = "models/plant_model_final.h5"   # change if your name differs
CLASS_NAMES_PATH = "models/class_indices.json"
IMG_SIZE = (160,160)

# ---------------- LOAD MODEL ---------------- #
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- LOAD CLASS NAMES ---------------- #
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# If saved as index:name dictionary, convert properly
if isinstance(class_names, dict):
    class_names = [class_names[str(i)] for i in range(len(class_names))]

# ---------------- PREDICTION FUNCTION ---------------- #
def predict_image(img_path):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    import numpy as np

    # Load image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)

    # Expand batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # IMPORTANT: Use same preprocessing as training
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)

    # Get predicted class index
    class_index = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))

    # 🔍 DEBUG SECTION
    print("\nRaw prediction vector (first 10 values):")
    print(preds[0][:10])

    print("\nTop 5 predicted indices + probabilities:")
    top_5 = np.argsort(preds[0])[-5:][::-1]
    for i in top_5:
        print(f"Index {i} → Probability: {preds[0][i]:.6f}")

    print("\nPredicted index:", class_index)

    return class_names[class_index], confidence

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    label, conf = predict_image(image_path)

    print("\n🌿 Prediction Result:")
    print("📌 Disease/Class:", label)
    print("🎯 Confidence:", round(conf * 100, 2), "%")

