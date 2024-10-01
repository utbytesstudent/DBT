# Detta är en Demo för MoveNet Lightning
import tensorflow as tf
import numpy as np
import cv2
import time

# Function to draw keypoints on a frame
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

# Edges to connect keypoints with lines
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c', (0, 5): 'm', (0, 6): 'c',
    (5, 7): 'm', (7, 9): 'm', (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm', (12, 14): 'c', (14, 16): 'c'
}

# Function to draw connections between keypoints
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# Load the MoveNet model (replace with your own model path)
interpreter = tf.lite.Interpreter(model_path=r'C:\Users\Zacharina\Downloads\movenet-tflite-singlepose-lightning-v1\3.tflite')
interpreter.allocate_tensors()

# Replace with the path to your video file
video_path = r"C:\Users\Zacharina\Videos\test.MP4"  # Change this to your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the width, height, and FPS of the original video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Ask the user for the output filename
output_filename = "movenetlite.avi"

# Define codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change codec if necessary
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Initialize variables for calculating FPS
prev_frame_time = 0
new_frame_time = 0

# Frame-by-frame video processing loop
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  # Exit the loop if no frame is captured (end of video)

    # Resize and preprocess the frame for the model
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    # Set up input and output details for the model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run the model to get keypoints
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # Render the skeleton on the frame
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)

    # Calculate FPS
    new_frame_time = time.time()
    fps_value = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time

    # Convert FPS to an integer and display it on the frame
    fps_text = f'FPS: {int(fps_value)}'
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Save the current frame to the output video
    out.write(frame)

# Release the video capture, video writer, and close all OpenCV windows
cap.release()
out.release()
