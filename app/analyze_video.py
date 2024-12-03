import cv2
import os
import glob
import torch
import torchvision.models as models
from face_detection import detect_faces  # Placeholder for face detection function
from object_detection import detect_objects  # Placeholder for object detection function
from scene_recognition import detect_scene_changes, extract_key_frames  # Placeholder for scene recognition functions
from audio_analysis import analyze_audio  # Placeholder for audio analysis function
from flask import Flask, jsonify, send_from_directory, render_template_string
import json
import threading
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import logging
from utils import reencode_video  # Import the re-encode function

# Suppress specific warnings related to H.264 codec
logging.getLogger("cv2").setLevel(logging.ERROR)

def find_video_file(video_dir):
    # Search for any video file in the specified directory
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']  # List of video extensions to look for
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))  # Search for video files matching extensions
    
    return video_files

def collect_new_data(frame, faces, objects, scenes, audio):
    """
    Collect new data for retraining the model.
    :param frame: The current video frame.
    :param faces: Detected faces in the frame.
    :param objects: Detected objects in the frame.
    :param scenes: Detected scenes in the frame.
    :param audio: Analyzed audio features.
    :return: A tuple of the frame and its annotations.
    """
    return (frame, faces, objects, scenes, audio)

def retrain_model(new_data, model, device, num_epochs=5):
    """
    Retrain the model with new data.
    :param new_data: New data to retrain the model.
    :param model: The model to retrain.
    :param device: The device to use for training (CPU or GPU).
    :param num_epochs: Number of epochs to train the model.
    """
    # Define a simple dataset and dataloader for the new data
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    new_dataset = SimpleDataset(new_data)
    new_dataloader = torch.utils.data.DataLoader(new_dataset, batch_size=32, shuffle=True)

    # Define a simple training loop
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in new_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(new_dataloader)}")

    # Save the updated model weights
    torch.save(model.state_dict(), model_path)
    print("Model retrained and saved.")

def analyze_video(input_video_path, model, device):
    # Re-encode the video using FFmpeg to ensure proper formatting
    reencoded_video_path = input_video_path.replace('.mp4', '_reencoded.mp4')
    reencode_video(input_video_path, reencoded_video_path)

    # Open the re-encoded video and process it frame by frame
    cap = cv2.VideoCapture(reencoded_video_path)

    # Check if video was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {reencoded_video_path}.")
        return
    
    frame_count = 0
    output_folder = '/app/output_results'
    os.makedirs(output_folder, exist_ok=True)

    new_data = []  # Collect new data for retraining

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Process frame (you can add more analysis here)
        print(f"Processing frame {frame_count}")

        # Example: Detect faces, objects, and scenes in each frame
        faces = detect_faces(frame)
        objects = detect_objects(frame)
        scenes = detect_scene_changes(input_video_path)
        audio = analyze_audio(input_video_path)

        # Collect new data for retraining
        new_data.append(collect_new_data(frame, faces, objects, scenes, audio))

        # Print the detected faces, objects, and scenes
        print(f"Detected faces: {faces}")
        print(f"Detected objects: {objects}")
        print(f"Detected scenes: {scenes}")
        print(f"Detected audio: {audio}")

        # Draw the face locations and names on the frame for visualization
        for (top, right, bottom, left), name in zip(faces[0], faces[1]):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Draw the object locations on the frame for visualization
        for obj in objects:
            x, y, w, h = obj['box']
            label = obj['label']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Object {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the frame with detections to the output folder
        frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)

    cap.release()
    print(f"Finished processing {frame_count} frames.")

    # Retrain the model with the new data
    retrain_model(new_data, model, device)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_model(model, device, train_loader, num_epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()  # Initialize GradScaler for mixed precision training
    metrics = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast():  # Use autocast for mixed precision training
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        metrics.append({'epoch': epoch + 1, 'loss': epoch_loss, 'accuracy': epoch_acc})
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Save metrics to a file
        with open('/app/output_results/training_metrics.json', 'w') as f:
            json.dump(metrics, f)

    print("Training completed.")

def main():
    # Define the directory where video files are located
    video_dir = '/app/input_videos/'
    saved_model_dir = '/app/saved_ai/'
    model_path = os.path.join(saved_model_dir, 'model.pth')

    # Define your model architecture
    model = models.resnet18(pretrained=False)

    # Check if a saved model exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Model loaded from", model_path)
    else:
        print("No saved model found. Starting training from scratch.")

    # Check if GPU is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Find video files in the directory
    video_files = find_video_file(video_dir)

    if not video_files:
        print("No video files found in the input_videos directory.")
    else:
        # Re-encode and process each video file found
        for video_file in video_files:
            print(f"Found video file: {video_file}")
            analyze_video(video_file, model, device)

        # Save the trained model
        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)
        torch.save(model.state_dict(), model_path)
        print("Model saved to", model_path)

    # Train the model
    print("Starting model training...")
    # Placeholder for train_loader, replace with actual data loader
    train_data = []  # Replace with actual data
    train_dataset = CustomDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)  # Use multiple workers and pinned memory for data loading
    train_model(model, device, train_loader)
    print("Model training finished.")

    # Save the trained model
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    torch.save(model.state_dict(), model_path)
    print("Model saved to", model_path)

def run_flask():
    app = Flask(__name__)

    @app.route('/')
    def index():
        image_folder = '/app/output_results'
        images = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        images.sort()  # Sort images by name
        return render_template_string('''
            <!doctype html>
            <title>AI Training Visualization</title>
            <h1>AI Training Visualization</h1>
            <div>
                {% for image in images %}
                    <img src="{{ url_for('image', filename=image) }}" style="width: 100%; max-width: 600px; margin-bottom: 20px;">
                {% endfor %}
            </div>
            <h2>Training Metrics</h2>
            <div id="metrics"></div>
            <script>
                async function fetchMetrics() {
                    const response = await fetch('/metrics');
                    const metrics = await response.json();
                    const metricsDiv = document.getElementById('metrics');
                    metricsDiv.innerHTML = '<pre>' + JSON.stringify(metrics, null, 2) + '</pre>';
                }
                fetchMetrics();
            </script>
        ''', images=images)

    @app.route('/images/<filename>')
    def image(filename):
        return send_from_directory('/app/output_results', filename)

    @app.route('/metrics')
    def metrics():
        metrics_file = '/app/output_results/training_metrics.json'
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return jsonify(metrics)
        else:
            return jsonify({"error": "Metrics file not found"}), 404

    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    # Start the Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Run the main function
    main()