from PIL import Image
import os

# Path to your dataset
dataset_path = r"C:\Users\DELL\dance_judge_AI\dataset"

# Convert all .jfif images in subfolders
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(".jfif"):
            jfif_path = os.path.join(root, file)
            jpg_path = os.path.splitext(jfif_path)[0] + ".jpg"

            # Open and convert the image
            with Image.open(jfif_path) as img:
                img.convert("RGB").save(jpg_path, "JPEG")

            print(f"Converted: {jfif_path} → {jpg_path}")

            # Optionally, remove the original .jfif file
            os.remove(jfif_path)
