import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ✅ Model path
model_path = "C:/Users/DELL/dance_judge_AI/costume_model_v3.h5"
video_path ="C:/Users/DELL/Desktop/test dance/videoplayback (4).mp4"   # Update with your video path

# ✅ Load model
model = load_model(model_path)
print("[INFO] Model loaded successfully.")

# ✅ Class names
class_names = ["bharatanatyam", "nonbharatanatyam"]

# ✅ Parameters
img_size = (150, 150)
frame_interval = 30  # Process every 30th frame
min_confidence_threshold = 0.75  # Set confidence threshold to prevent misclassifications

# ✅ Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("[ERROR] Cannot open video.")
    exit()

frame_count = 0
bharatanatyam_confidence = []
nonbharatanatyam_confidence = []

# ✅ Frame processing loop
while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # ✅ Preprocess frame
        resized_frame = cv2.resize(frame, img_size)
        img = img_to_array(resized_frame) / 255.0
        img = np.expand_dims(img, axis=0)

        # ✅ Predict
        probabilities = model.predict(img, verbose=0)[0]

        # ✅ Handle binary classification
        if len(probabilities) == 1:
            bharatanatyam_prob = 1 - probabilities[0]
            nonbharatanatyam_prob = probabilities[0]
        else:
            bharatanatyam_prob = probabilities[0]
            nonbharatanatyam_prob = probabilities[1]

        # ✅ Display probabilities
        print(f"[INFO] Frame {frame_count}: Bharatanatyam={bharatanatyam_prob*100:.2f}%, "
              f"Non-Bharatanatyam={nonbharatanatyam_prob*100:.2f}%")

        # ✅ Apply threshold and append probabilities
        if max(bharatanatyam_prob, nonbharatanatyam_prob) >= min_confidence_threshold:
            bharatanatyam_confidence.append(bharatanatyam_prob)
            nonbharatanatyam_confidence.append(nonbharatanatyam_prob)

    frame_count += 1

# ✅ Release video
cap.release()

# ✅ Calculate final score
if len(bharatanatyam_confidence) > 0 and len(nonbharatanatyam_confidence) > 0:
    avg_bharatanatyam_score = np.mean(bharatanatyam_confidence) * 100
    avg_nonbharatanatyam_score = np.mean(nonbharatanatyam_confidence) * 100

    # ✅ Proper scoring logic
    if avg_bharatanatyam_score > avg_nonbharatanatyam_score:
        final_costume_score = avg_bharatanatyam_score
    else:
        final_costume_score = 100 - avg_nonbharatanatyam_score

    print(f"\n[INFO] Final Costume Score for Video: {final_costume_score:.2f}%")
else:
    print("\n[INFO] No valid frames processed.")
