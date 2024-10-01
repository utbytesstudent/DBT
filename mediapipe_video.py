import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video Feed (replace this path with your video file path)
cap = cv2.VideoCapture(r'C:\Users\Zacharina\Videos\test.MP4')

# Get the width, height, and frame rate of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Fallback FPS if not properly retrieved from the video
if fps == 0:
    fps = 30  # Use 30 as a default

# Define codec and create VideoWriter object to save output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('mediapipe.avi', fourcc, fps, (frame_width, frame_height))

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    prev_time = 0  # Initialize time tracking for FPS calculation

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        # Start time measurement for this frame
        start_time = time.time()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # End time measurement for this frame
        end_time = time.time()

        # Calculate FPS based on time difference between frames
        frame_time = end_time - start_time
        current_fps = 1 / frame_time if frame_time > 0 else 0

        # Convert FPS to an integer and display it on the frame
        fps_text = f'FPS: {int(current_fps)}'
        cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write the frame into the file
        out.write(image)

# Release resources
cap.release()
out.release()
