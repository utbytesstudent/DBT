import cv2
import mediapipe as mp

# Initialize MediaPipe Pose model and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define function to process a video and apply skeleton tracking
def process_video(input_path, output_path):
    # Initialize video capture and writer
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object to save the output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize MediaPipe Pose solution
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the image from BGR (OpenCV format) to RGB (MediaPipe format)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False  # Improves performance

            # Process the image and detect poses
            results = pose.process(image_rgb)

            # Draw the pose annotation on the image
            image_rgb.flags.writeable = True  # Re-enable the writeability
            if results.pose_landmarks:
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Landmarks
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # Connections
                )

            # Write the annotated frame to the output video
            out.write(frame)

            # Optional: Display the frame with annotations
            cv2.imshow('Skeleton Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Input and output paths
input_video = r'C:\Users\Zacharina\Desktop\goprovideor\GX010941.MP4'
output_video = '41_2.avi'

# Call the function to process the video
process_video(input_video, output_video)
