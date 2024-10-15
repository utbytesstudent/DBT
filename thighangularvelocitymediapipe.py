import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose 

def calculate_angle(hip, knee):
    """
    Calculate the angle of the thigh (hip to knee) with respect to the horizontal axis.
    Arguments:
    hip -- (x, y) coordinates of the hip joint
    knee -- (x, y) coordinates of the knee joint
    
    Returns:
    angle -- the angle of the thigh with respect to the horizontal axis (in radians)
    """
    delta_y = knee[1] - hip[1]
    delta_x = knee[0] - hip[0]
    angle = np.arctan2(delta_y, delta_x)  # angle in radians
    return angle

def calculate_thigh_angular_velocity(hip_coords, knee_coords, time_stamps):
    """
    Calculate the angular velocity of the thigh over time.
    
    Arguments:
    hip_coords -- list of (x, y) tuples representing the hip joint coordinates over time
    knee_coords -- list of (x, y) tuples representing the knee joint coordinates over time
    time_stamps -- list of time stamps corresponding to each frame
    
    Returns:
    angular_velocities -- list of angular velocities (in radians/second)
    """
    angular_velocities = []
    
    for i in range(1, len(hip_coords)):
        # Calculate the thigh angle at the current and previous frames
        angle_prev = calculate_angle(hip_coords[i-1], knee_coords[i-1])
        angle_curr = calculate_angle(hip_coords[i], knee_coords[i])
        
        # Compute the time difference between frames
        delta_t = time_stamps[i] - time_stamps[i-1]
        
        # Calculate the angular velocity (change in angle over time)
        angular_velocity = (angle_curr - angle_prev) / delta_t
        angular_velocities.append(angular_velocity)
    
    return angular_velocities

# Input video file
input_video_path = r'C:\Users\Zacharina\Desktop\goprovideor\GX010938.mp4'
output_video_path = r'C:\Users\Zacharina\Desktop\goprovideor\mediapipe_output_video_with_thigh_angular_velocity.mp4'

cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up video writer to save the output
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    hip_coords = []
    knee_coords = []
    time_stamps = []
    start_time = time.time()  # For calculating time difference between frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there's an issue with the frame capture
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for hip and knee
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame_width,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame_height]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame_width,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame_height]
            
            # Store the coordinates and timestamp
            hip_coords.append(hip)
            knee_coords.append(knee)
            time_stamps.append(time.time() - start_time)
            
        except:
            pass
        
        # Render pose landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        # Write the frame with pose landmarks to the output video
        out.write(image)

    # Once all frames are processed, calculate the thigh angular velocity
    if len(hip_coords) > 1 and len(knee_coords) > 1:
        angular_velocities = calculate_thigh_angular_velocity(hip_coords, knee_coords, time_stamps)
        print("Thigh Angular Velocities (in radians/second):", angular_velocities)

cap.release()
out.release()
cv2.destroyAllWindows()
