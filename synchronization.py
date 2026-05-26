import cv2
import numpy as np
import mediapipe as mp
import os

def extract_landmarks(frame, pose):
    """Extract pose landmarks from a frame."""
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
    return None

def calculate_sync_score(landmarks1, landmarks2):
    """Calculate synchronization score between two sets of landmarks."""
    if landmarks1 is None or landmarks2 is None:
        return 0.0
    
    dist = np.linalg.norm(landmarks1 - landmarks2)
    max_dist = np.sqrt(3) * len(landmarks1)
    score = max(0, 100 - (dist / max_dist) * 100)
    return score

def process_video(video_path):
    """Process local video file and display synchronization score only."""
    if not os.path.isfile(video_path):
        print(f"Error: File not found - {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        mid_x = width // 2

        # Split the frame into two halves (dancer1 and dancer2)
        dancer1_frame = frame[:, :mid_x]
        dancer2_frame = frame[:, mid_x:]

        # Extract landmarks for both dancers
        landmarks1 = extract_landmarks(dancer1_frame, pose)
        landmarks2 = extract_landmarks(dancer2_frame, pose)

        # Calculate synchronization score
        score = calculate_sync_score(landmarks1, landmarks2)
        scores.append(score)

        # Display the frame with score
        cv2.putText(frame, f"Sync Score: {score:.2f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Dance Sync Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Display final synchronization score in the terminal only
    if scores:
        final_score = np.mean(scores)
        print(f"Final Synchronization Score: {final_score:.2f}%")
    else:
        print("No frames were processed.")

# Example: Use local video paths
video_path = "C://Users/DELL/Desktop/test dance/videoplayback (3).mp4"  # Replace with your local video path
process_video(video_path)
