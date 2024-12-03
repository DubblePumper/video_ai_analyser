import cv2
import numpy as np
import os
import tempfile
from datetime import datetime

def detect_scene_changes(video_path, threshold=30):
    """
    Detect scene changes in a video by comparing consecutive frames.
    :param video_path: Path to the video file.
    :param threshold: Threshold for detecting scene changes based on frame difference.
    :return: List of frame indices where scene changes occur.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        scene_changes = []
        ret, prev_frame = cap.read()

        if not ret:
            print(f"Error reading video: {video_path}")
            return []

        # Convert to grayscale
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_count = 0
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            # Convert current frame to grayscale
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Calculate absolute difference between consecutive frames
            frame_diff = cv2.absdiff(curr_frame, prev_frame)
            non_zero_count = np.count_nonzero(frame_diff)

            # If the number of changed pixels is above the threshold, mark as scene change
            if non_zero_count > threshold:
                scene_changes.append(frame_count)

            prev_frame = curr_frame
            frame_count += 1

        cap.release()
        return scene_changes
    except Exception as e:
        print(f"Error detecting scene changes: {e}")
        return []

def extract_key_frames(video_path, scene_changes, output_folder="app/output_results"):
    """
    Extract key frames from the video at the points where scene changes occur.
    :param video_path: Path to the video file.
    :param scene_changes: List of frame indices where scene changes occur.
    :param output_folder: Folder where key frames will be saved.
    """
    try:
        cap = cv2.VideoCapture(video_path)

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        for idx, frame_index in enumerate(scene_changes):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # Set the frame position

            ret, frame = cap.read()
            if not ret:
                continue

            # Save the key frame as an image
            frame_filename = os.path.join(output_folder, f"key_frame_{idx + 1}_{frame_index}.jpg")
            cv2.imwrite(frame_filename, frame)

        cap.release()
    except Exception as e:
        print(f"Error extracting key frames: {e}")

def analyze_scene_transitions(video_path):
    """
    Main function to detect scene changes and extract key frames.
    :param video_path: Path to the video file.
    :return: A dictionary with scene change frames and extracted key frames.
    """
    # Detect scene changes
    scene_changes = detect_scene_changes(video_path)

    # Extract key frames at scene change points
    extract_key_frames(video_path, scene_changes)

    # Return results
    return {
        "scene_changes": scene_changes,
        "key_frames_extracted": len(scene_changes)
    }

def process_all_videos_in_folder(folder_path="app/input_videos"):
    """
    Automatically process all video files in a folder for scene changes.
    :param folder_path: Path to the folder containing video files.
    :return: A dictionary with scene changes and key frame extraction results for each video.
    """
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    results = {}
    
    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        print(f"Processing video for scene recognition: {video_path}")

        video_results = analyze_scene_transitions(video_path)
        results[video_file] = video_results

    return results

# Example usage: Process all videos in the input_videos folder
folder_path = "app/input_videos"  # Replace with your folder path if different
scene_analysis_results = process_all_videos_in_folder(folder_path)

# Print scene analysis results
for video, analysis in scene_analysis_results.items():
    print(f"Results for {video}:")
    print(f"Scene Changes Detected: {analysis['scene_changes']}")
    print(f"Key Frames Extracted: {analysis['key_frames_extracted']}")