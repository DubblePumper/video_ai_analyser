import cv2
import numpy as np
import os

# Load the pre-trained YOLO model (you can also use other models such as SSD or Faster R-CNN)
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getLayers()]
    return net, output_layers

def detect_objects_in_frame(frame, net, output_layers, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Detect objects in a single video frame.
    :param frame: The current video frame.
    :param net: The YOLO model network.
    :param output_layers: The output layers of the YOLO model.
    :param confidence_threshold: Minimum confidence for object detection.
    :param nms_threshold: Non-Maximum Suppression threshold to remove duplicate boxes.
    :return: List of bounding boxes and labels of detected objects.
    """
    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Initialize lists for bounding boxes, confidences, and class ids
    boxes, confidences, class_ids = [], [], []
    height, width, _ = frame.shape

    # Process each detected object
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                # Get the center coordinates and width/height of the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates for bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Append to lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to remove duplicate boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detected_objects.append({"label": class_ids[i], "box": (x, y, w, h)})

    return detected_objects

def process_video_for_object_detection(video_path, output_folder="app/output_results"):
    """
    Process a video file and detect objects in each frame.
    :param video_path: Path to the video file.
    :param output_folder: Folder to save frames with detected objects.
    :return: A list of results containing detected objects for each frame.
    """
    # Load YOLO model
    net, output_layers = load_yolo_model()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_results = []

    # Process each frame of the video
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the current frame
        detected_objects = detect_objects_in_frame(frame, net, output_layers)

        # Save frame with bounding boxes drawn on detected objects
        for obj in detected_objects:
            x, y, w, h = obj['box']
            label = obj['label']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Object {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the processed frame with bounding boxes to the output folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_results.append({
            "frame": frame_count,
            "objects_detected": detected_objects
        })
        frame_count += 1

    cap.release()
    return frame_results

def process_all_videos_in_folder(folder_path="app/input_videos"):
    """
    Automatically process all video files in a folder for object detection.
    :param folder_path: Path to the folder containing video files.
    :return: A dictionary with object detection results for each video.
    """
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    results = {}

    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f"Processing video for object detection: {video_path}")

        video_results = process_video_for_object_detection(video_path)
        results[video_file] = video_results

    return results

# Example usage: Process all videos in the input_videos folder
folder_path = "app/input_videos"  # Replace with your folder path if different
object_detection_results = process_all_videos_in_folder(folder_path)

# Print object detection results
for video, frames in object_detection_results.items():
    print(f"Results for {video}:")
    for frame_result in frames:
        print(f"Frame {frame_result['frame']} detected {len(frame_result['objects_detected'])} objects.")