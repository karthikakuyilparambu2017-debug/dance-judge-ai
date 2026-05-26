import cv2
import mediapipe as mp
import numpy as np

# ✅ Path to your local dance video
video_path = r"C:/Users/DELL/Desktop/test dance/videoplayback (2).mp4"

# ✅ Initialize face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# ✅ Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# ✅ Variables to track makeup evaluation
total_makeup_score = 0
face_count = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        face_count += 1

        # ✅ Analyze makeup based on bounding box size and detection confidence
        for detection in results.detections:
            score = detection.score[0]  # Confidence score
            makeup_score = min(100, max(0, score * 100))  # Normalize to 0-100%
            total_makeup_score += makeup_score

# ✅ Calculate the final makeup score
if face_count > 0:
    final_makeup_score = total_makeup_score / face_count
else:
    final_makeup_score = 0.0

# ✅ Display only the final score in the terminal
print(f"Final Makeup Score: {final_makeup_score:.2f}%")

# ✅ Release resources
cap.release()
cv2.destroyAllWindows()
