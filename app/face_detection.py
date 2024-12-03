import cv2
import face_recognition
import numpy as np  # Correct the import statement
import os
import logging
import torch  # Voeg deze import toe

# Initialiseer variabelen voor bekende gezichtsencoderingen en namen
known_face_encodings = []
known_face_names = []

# Configureer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Functie om gezichten te detecteren en te herkennen, en deze dynamisch toe te voegen aan de lijst van bekende gezichten
def detect_faces(frame):
    global known_face_encodings, known_face_names

    # Convert the frame from BGR (OpenCV format) to RGB (Face Recognition format)
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])  # Correct the variable name

    # Detect faces and their encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = []

    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = rgb_frame[top:bottom, left:right]
        face_landmarks = face_recognition.face_landmarks(face_image)
        if face_landmarks:
            # Correct the function call by passing the correct arguments
            face_encoding = face_recognition.face_encodings(rgb_frame, known_face_locations=[face_location])[0]
            face_encodings.append(face_encoding)

    # Initialize an empty list for recognized faces
    recognized_faces = []

    for face_encoding in face_encodings:
        # Check if this face matches any of the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If there's a match, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            # If no match, add this new face encoding and associate it with a name
            known_face_encodings.append(face_encoding)
            name = f"Person {len(known_face_names) + 1}"
            known_face_names.append(name)

        recognized_faces.append(name)

    return face_locations, recognized_faces

def count_people(frame):
    """
    Tel het aantal gezichten dat in het frame gedetecteerd is.
    """
    face_locations, _ = detect_faces(frame)
    return len(face_locations)

def analyze_face_details(frame):
    """
    Analyseer het frame voor gezichtsherkenning en details (zoals leeftijd, geslacht en emotie).
    """
    face_locations, recognized_faces = detect_faces(frame)

    # Initialiseer dictionary om de details op te slaan
    people_details = {}

    for (top, right, bottom, left), name in zip(face_locations, recognized_faces):
        # Crop het gezicht voor verdere analyse
        face_image = frame[top:bottom, left:right]

        # Optioneel: Verwerk het cropped gezicht voor extra details (leeftijd, geslacht, emotie)
        # Voor nu gebruiken we tijdelijke waarden
        age = "Unknown"
        gender = "Unknown"
        emotion = "Unknown"

        # Sla de details op voor het herkende gezicht
        people_details[name] = {
            "location": (top, right, bottom, left),
            "age": age,
            "gender": gender,
            "emotion": emotion
        }

    # Tel het aantal mensen dat gedetecteerd is
    people_count = len(face_locations)

    return people_count, people_details

def analyze_video(frame):
    """
    Analyseer het frame voor gezichtsherkenning en details.
    """
    # Analyseer gezichtsdetails en verkrijg het aantal mensen en details
    people_count, people_details = analyze_face_details(frame)

    # Combineer de resultaten
    results = {
        "people_count": people_count,
        "people_details": people_details
    }

    return results

# Functie om het laatst verwerkte frame op te slaan
def save_last_processed_frame(video_file, frame_number):
    with open(f"{video_file}_last_frame.txt", "w") as f:
        f.write(str(frame_number))

# Functie om het laatst verwerkte frame te laden
def load_last_processed_frame(video_file):
    try:
        with open(f"{video_file}_last_frame.txt", "r") as f:
            data = f.read().strip()
            if data:
                return int(data)
            else:
                return 0
    except (FileNotFoundError, ValueError):
        return 0

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

input_folder = 'input_videos'  # vervang met jouw pad naar de input folder
saved_model_dir = '/app/saved_ai/'
model_path = os.path.join(saved_model_dir, 'model.pth')

# Zorg ervoor dat de directory voor het opgeslagen model bestaat
os.makedirs(saved_model_dir, exist_ok=True)

model = torch.nn.Module()  # Definieer hier jouw model

for video_file in os.listdir(input_folder):
    if video_file.endswith(('.mp4', '.avi', '.mov')):  # Voeg andere videoformaten toe indien nodig
        video_path = os.path.join(input_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = load_last_processed_frame(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Zet de video capture op de laatst verwerkte frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        new_data = []  # Collect new data for retraining

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            logging.info(f"Verwerken frame {frame_count}/{total_frames} van video {video_file}")

            # Analyseer het huidige frame voor gezichtsherkenning en mensen tellen
            details = analyze_video(frame)

            # Collect new data for retraining
            new_data.append((frame, details))

            # Toon de details (je kunt de details ook opslaan indien nodig)
            print(details)

            # Teken de gezichtsposities en namen op het frame voor visualisatie
            for name, info in details["people_details"].items():
                top, right, bottom, left = info["location"]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # Toon bijvoorbeeld extra details zoals leeftijd, geslacht, emotie
                face_info = f"Leeftijd: {info['age']}, Geslacht: {info['gender']}, Emotie: {info['emotion']}"
                cv2.putText(frame, face_info, (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Sla het model en het laatst verwerkte frame op elke 100 frames
            if frame_count % 100 == 0:
                torch.save(model.state_dict(), model_path)
                save_last_processed_frame(video_file, frame_count)
                logging.info(f"Model opgeslagen en laatst verwerkte frame bijgewerkt na {frame_count} frames.")

        # Sla het laatst verwerkte frame op wanneer de videoverwerking is voltooid
        save_last_processed_frame(video_file, frame_count)
        cap.release()

        # Retrain the model with the new data
        retrain_model(new_data, model, device)