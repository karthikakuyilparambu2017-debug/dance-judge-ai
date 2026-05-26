import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ✅ Load the model
model_path = "C:/Users/DELL/dance_judge_AI/costume_model_v3.h5"
model = load_model(model_path)

# ✅ Test images folder
bharatanatyam_folder ="C:/Users/DELL/dance_judge_AI/dataset/bharatanatyam"
nonbharatanatyam_folder ="C:/Users/DELL/dance_judge_AI/dataset/nonbharatanatyam"
# ✅ Function to preprocess image
def preprocess_image(image_path, img_size=(150, 150)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ✅ Function to verify predictions
def verify_predictions(folder, label):
    incorrect_predictions = 0
    total_predictions = 0
    confident_misclassifications = 0

    print(f"\n[INFO] Testing: {label}")

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            continue

        img = preprocess_image(img_path)
        probabilities = model.predict(img, verbose=0)[0]

        # ✅ Handle binary model
        if len(probabilities) == 1:
            bharatanatyam_prob = 1 - probabilities[0]
            nonbharatanatyam_prob = probabilities[0]
        else:
            # ✅ Handle categorical model
            bharatanatyam_prob = probabilities[0]
            nonbharatanatyam_prob = probabilities[1]

        total_predictions += 1

        # ✅ Check for incorrect predictions
        correct_class = "Bharatanatyam" if label == "bharatanatyam" else "Non-Bharatanatyam"
        predicted_class = "Bharatanatyam" if bharatanatyam_prob > nonbharatanatyam_prob else "Non-Bharatanatyam"

        if predicted_class != correct_class:
            incorrect_predictions += 1
            if max(bharatanatyam_prob, nonbharatanatyam_prob) > 0.90:
                confident_misclassifications += 1

        print(f"[INFO] {img_name}: Bharatanatyam={bharatanatyam_prob*100:.2f}%, "
              f"Non-Bharatanatyam={nonbharatanatyam_prob*100:.2f}% → Predicted: {predicted_class}")

    # ✅ Display results
    print("\n[RESULTS]")
    print(f"Total Images Tested: {total_predictions}")
    print(f"Incorrect Predictions: {incorrect_predictions}")
    print(f"Confident Misclassifications (>90% confidence): {confident_misclassifications}")

# ✅ Run verification for both classes
verify_predictions(bharatanatyam_folder, "bharatanatyam")
verify_predictions(nonbharatanatyam_folder, "nonbharatanatyam")
