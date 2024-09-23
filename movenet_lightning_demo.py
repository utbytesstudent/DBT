# Detta är en Demo för MoveNet Lightning
import tensorflow as tf
import numpy as np
import cv2

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
video_path =r"C:\Users\Zacharina\Pictures\Camera Roll\testvideo.mp4"  # Change this to your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

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
    
    # Show the frame with skeleton overlay
    cv2.imshow('MoveNet Lightning - Skeleton Tracking', frame)
    
    # Press 'q' to exit the video
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
