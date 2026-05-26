import cv2 
import numpy as np
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp

# ✅ Check if video path is provided as argument
if len(sys.argv) < 2:
    print("[ERROR] No video file provided.")
    sys.exit(1)

# ✅ Use the provided video path
video_path = sys.argv[1]

# ✅ Paths
model_path = "C:/Users/DELL/dance_judge_AI/costume_model_v3.h5"

# ✅ Load costume model
model = load_model(model_path)
print("[INFO] Costume model loaded successfully.")

# ✅ Mediapipe for pose and face detection
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
pose = mp_pose.Pose()
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# ✅ Parameters
img_size = (150, 150)
frame_interval = 30
min_confidence_threshold = 0.75

# ✅ Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {video_path}")
    sys.exit(1)

# ✅ Variables for scores
frame_count = 0
synchronization_scores = []
bharatanatyam_confidence = []
nonbharatanatyam_confidence = []
total_makeup_score = 0
face_count = 0

# ✅ Function to extract landmarks
def extract_landmarks(frame, pose):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
    return None

# ✅ Function to calculate synchronization score
def calculate_sync_score(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return 0.0
    dist = np.linalg.norm(landmarks1 - landmarks2)
    max_dist = np.sqrt(3) * len(landmarks1)
    score = max(0, 100 - (dist / max_dist) * 100)
    return score

# ✅ Process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, _ = frame.shape
    mid_x = width // 2
    
    # ✅ Split frame into two halves
    dancer1_frame = frame[:, :mid_x]
    dancer2_frame = frame[:, mid_x:]
    
    # ✅ Pose landmarks for synchronization
    landmarks1 = extract_landmarks(dancer1_frame, pose)
    landmarks2 = extract_landmarks(dancer2_frame, pose)
    sync_score = calculate_sync_score(landmarks1, landmarks2)
    synchronization_scores.append(sync_score)
    
    # ✅ Face detection for makeup evaluation
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detection.process(rgb_frame)
    if face_results.detections:
        face_count += 1
        for detection in face_results.detections:
            makeup_score = min(100, max(0, detection.score[0] * 100))
            total_makeup_score += makeup_score
    
    # ✅ Costume evaluation every N frames
    if frame_count % frame_interval == 0:
        resized_frame = cv2.resize(frame, img_size)
        img = img_to_array(resized_frame) / 255.0
        img = np.expand_dims(img, axis=0)
        probabilities = model.predict(img, verbose=0)[0]
        
        if len(probabilities) == 1:
            bharatanatyam_prob = 1 - probabilities[0]
            nonbharatanatyam_prob = probabilities[0]
        else:
            bharatanatyam_prob = probabilities[0]
            nonbharatanatyam_prob = probabilities[1]
        
        if max(bharatanatyam_prob, nonbharatanatyam_prob) >= min_confidence_threshold:
            bharatanatyam_confidence.append(bharatanatyam_prob)
            nonbharatanatyam_confidence.append(nonbharatanatyam_prob)
    
    # ✅ Display live scores
    cv2.putText(frame, f"Sync: {sync_score:.2f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Dance Judge AI', frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release resources
cap.release()
cv2.destroyAllWindows()

# ✅ Calculate final scores
final_sync_score = np.mean(synchronization_scores) if synchronization_scores else 0.0
final_makeup_score = (total_makeup_score / face_count) if face_count > 0 else 0.0

if bharatanatyam_confidence and nonbharatanatyam_confidence:
    avg_bharatanatyam_score = np.mean(bharatanatyam_confidence) * 100
    avg_nonbharatanatyam_score = np.mean(nonbharatanatyam_confidence) * 100
    final_costume_score = avg_bharatanatyam_score if avg_bharatanatyam_score > avg_nonbharatanatyam_score else 100 - avg_nonbharatanatyam_score
else:
    final_costume_score = 0.0

# ✅ Display final scores
print("\n[FINAL SCORES]")
print(f"Synchronization Score: {final_sync_score:.2f}%")
print(f"Makeup Score: {final_makeup_score:.2f}%")
print(f"Costume Score: {final_costume_score:.2f}%")
