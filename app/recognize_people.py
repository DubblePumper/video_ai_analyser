
import face_recognition
import os
import csv

def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(os.path.splitext(filename)[0])

    return known_face_encodings, known_face_names

def recognize_faces_in_image(image_path, known_face_encodings, known_face_names):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    recognized_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        recognized_names.append(name)

    return recognized_names

def process_images_and_recognize_faces(images_dir, known_faces_dir):
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_dir, filename)
            recognized_names = recognize_faces_in_image(image_path, known_face_encodings, known_face_names)
            print(f"Recognized in {filename}: {', '.join(recognized_names)}")

# Example usage
known_faces_dir = 'app/known_faces'
images_dir = 'app/input_images'
process_images_and_recognize_faces(images_dir, known_faces_dir)