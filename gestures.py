import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils


def main():
    # Load the video
    video = cv2.VideoCapture('gestures.mp4')

    # mediapipe for hands and shoulder
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize tracker for face
    tracker = cv2.TrackerKCF_create()

    # Read the first frame of the video
    ret, frame = video.read()

    # Detect the face in the first frame
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Initialize the tracker with the feature points
    bbox = faces[0]
    tracker.init(frame, bbox)
    
    # Track when a shrug happens
    prev_right_y = None
    prev_left_y = None

    # Loop over the video frames
    while True:
        # Read the next frame of the video
        ret, frame = video.read()
        if not ret:
            break

        # convert to RGB instead of BGR for mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand and pose(Shoulder) results
        hand_results = hands.process(frame_rgb)

        pose_results = pose.process(frame_rgb)

        # Shoulder detection
        if pose_results.pose_landmarks:
            left_shoulder_index = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            right_shoulder_index = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            
            left_shoulder = (int(pose_results.pose_landmarks.landmark[left_shoulder_index].x * frame.shape[1]), int(pose_results.pose_landmarks.landmark[left_shoulder_index].y * frame.shape[0]))
            right_shoulder = (int(pose_results.pose_landmarks.landmark[right_shoulder_index].x * frame.shape[1]), int(pose_results.pose_landmarks.landmark[right_shoulder_index].y * frame.shape[0]))
            
            # Calculate the width and height of a box that defines the shoulder lines
            box_width = abs(right_shoulder[0] - left_shoulder[0])
            box_height = int(box_width * 0.4)

            if prev_left_y and prev_right_y:
                # Calculate dy for each shoulder
                left_dy = pose_results.pose_landmarks.landmark[left_shoulder_index].y - prev_left_y
                right_dy = pose_results.pose_landmarks.landmark[right_shoulder_index].y - prev_right_y
                
                # If dy is above a threshold then change the color of the shoulder lines
                if abs(right_dy) > 0.0015 and abs(left_dy) > 0.0015:
                    cv2.line(frame, (left_shoulder[0] - box_width // 3, left_shoulder[1] - box_height // 2), (left_shoulder[0] + box_width // 4, left_shoulder[1] + box_height // 10), (255, 0, 255), 2)
                    cv2.line(frame, (right_shoulder[0] + box_width // 3, right_shoulder[1] - box_height // 2), (right_shoulder[0] - box_width // 4, right_shoulder[1] + box_height // 10), (255, 0, 255), 2)
                else:
                    cv2.line(frame, (left_shoulder[0] - box_width // 3, left_shoulder[1] - box_height // 2), (left_shoulder[0] + box_width // 4, left_shoulder[1] + box_height // 10), (0, 255, 0), 2)
                    cv2.line(frame, (right_shoulder[0] + box_width // 3, right_shoulder[1] - box_height // 2), (right_shoulder[0] - box_width // 4, right_shoulder[1] + box_height // 10), (0, 255, 0), 2)
            else:
                cv2.line(frame, (left_shoulder[0] - box_width // 3, left_shoulder[1] - box_height // 2), (left_shoulder[0] + box_width // 4, left_shoulder[1] + box_height // 10), (0, 255, 0), 2)
                cv2.line(frame, (right_shoulder[0] + box_width // 3, right_shoulder[1] - box_height // 2), (right_shoulder[0] - box_width // 4, right_shoulder[1] + box_height // 10), (0, 255, 0), 2)
            prev_left_y = pose_results.pose_landmarks.landmark[left_shoulder_index].y
            prev_right_y = pose_results.pose_landmarks.landmark[right_shoulder_index].y


        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Extract the x and y coordinates of each landmark
                x = [int(lm.x * frame.shape[1]) for lm in hand_landmarks.landmark]
                y = [int(lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]
                # Compute the minimum and maximum x and y coordinates of the hand
                x_min, x_max = min(x), max(x)
                y_min, y_max = min(y), max(y)
                # Draw a rectangle around the hand
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Update the tracker
        success, bbox = tracker.update(frame)

        # Draw box around head
        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture and close all windows
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
