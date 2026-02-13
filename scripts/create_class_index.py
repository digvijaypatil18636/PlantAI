import os
import json

TRAIN_PATH = "data/train"
OUTPUT_PATH = "models/class_indices.json"

# Get class folder names
class_names = sorted([
    folder for folder in os.listdir(TRAIN_PATH)
    if os.path.isdir(os.path.join(TRAIN_PATH, folder))
])

# Create index mapping
class_indices = {i: name for i, name in enumerate(class_names)}

# Save to JSON
with open(OUTPUT_PATH, "w") as f:
    json.dump(class_indices, f, indent=4)

print("✅ class_indices.json created successfully!")

