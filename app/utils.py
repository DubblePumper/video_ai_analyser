import cv2
import os
import shutil
import numpy as np
import subprocess  # Add subprocess for running FFmpeg commands
from moviepy.editor import VideoFileClip

def create_directory_if_not_exists(directory):
    """
    Create a directory if it doesn't already exist.
    :param directory: Directory path to check and create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_frame_to_file(frame, frame_count, output_folder):
    """
    Save a video frame to a file.
    :param frame: The frame to save.
    :param frame_count: The frame number.
    :param output_folder: Folder to save the frame.
    """
    create_directory_if_not_exists(output_folder)
    frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_filename, frame)

def get_video_files_from_folder(folder_path):
    """
    Get all video files from a folder.
    :param folder_path: Path to the folder containing video files.
    :return: List of video file paths.
    """
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

def process_video(video_path, output_folder):
    """
    Process the video by extracting frames and saving them.
    :param video_path: Path to the video file.
    :param output_folder: Folder where the frames will be saved.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        save_frame_to_file(frame, frame_count, output_folder)
        frame_count += 1
    cap.release()

def clean_up_old_results(output_folder):
    """
    Clean up any old result files in the output folder.
    :param output_folder: Folder where the results are saved.
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    create_directory_if_not_exists(output_folder)

def resize_frame(frame, width, height):
    """
    Resize a frame to the specified width and height.
    :param frame: The frame to resize.
    :param width: The new width of the frame.
    :param height: The new height of the frame.
    :return: The resized frame.
    """
    return cv2.resize(frame, (width, height))

def extract_faces_from_frame(frame, face_cascade):
    """
    Extract faces from a single video frame using Haar Cascade.
    :param frame: The frame from which faces will be extracted.
    :param face_cascade: The pre-trained Haar Cascade for face detection.
    :return: List of face bounding boxes (x, y, w, h).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def draw_bounding_boxes_on_frame(frame, boxes, color=(0, 255, 0)):
    """
    Draw bounding boxes on a frame.
    :param frame: The frame on which to draw bounding boxes.
    :param boxes: List of bounding boxes to draw [(x, y, w, h)].
    :param color: Color of the bounding box (default is green).
    """
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

def apply_threshold_on_frame(frame, threshold_value=100):
    """
    Apply a simple threshold to a frame to convert it to binary.
    :param frame: The frame to process.
    :param threshold_value: Threshold value for binary conversion.
    :return: The thresholded binary frame.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresholded_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_frame

def extract_keyframes_from_video(video_path, threshold=0.9):
    """
    Extract keyframes from a video where there is significant change.
    :param video_path: Path to the video.
    :param threshold: Threshold for detecting a significant change between frames.
    :return: List of keyframes.
    """
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    keyframes = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is None:
            prev_frame = frame
            continue

        # Compute the difference between the current and previous frames
        frame_diff = cv2.absdiff(prev_frame, frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

        # Calculate percentage of change
        change_percentage = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1])
        
        if change_percentage > threshold:
            keyframes.append(frame)

        prev_frame = frame

    cap.release()
    return keyframes

def reencode_video(input_video_path, output_video_path):
    """
    Re-encode a video file using FFmpeg to ensure proper formatting.
    :param input_video_path: Path to the input video file.
    :param output_video_path: Path to the output re-encoded video file.
    """
    try:
        command = [
            'ffmpeg', '-i', input_video_path, '-c:v', 'libx264', '-crf', '23', '-preset', 'fast', output_video_path
        ]
        subprocess.run(command, check=True)
        print(f"Re-encoded video saved to {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error re-encoding video: {e}")