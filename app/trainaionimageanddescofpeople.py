import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Configuratie
LEARNING_RATE = 0.001
BATCH_SIZE = 20
NUM_EPOCHS = 20
IMG_SIZE = 160  # VGGFace2 gebruikt 160x160 afbeeldingen
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENABLE_RANDOMIZATION = False  # Boolean to enable or disable randomization
NUM_PREDICTIONS = 50  # Global variable for the number of predictions

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paden
CSV_PATH = os.path.join(SCRIPT_DIR, 'datasets', 'dataset.csv')
IMG_DIR = os.path.join(SCRIPT_DIR, 'input_images')
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, 'saved_ai', 'person_recognition_model_vggface2.pth')
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Transformaties (meer randomisatie toegevoegd)
if (ENABLE_RANDOMIZATION):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        transforms.RandomAffine(30, shear=10),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisatie voor VGGFace2
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisatie voor VGGFace2
    ])

# Dataset
class PersonDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        print(f"Loading dataset from {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Filter alleen rijen met geldige afbeeldingen
        self.data = self.data[self.data['Name'].apply(self._has_image)]
        unique_names = self.data['Name'].unique()
        self.labels = {name: idx for idx, name in enumerate(unique_names)}
        self.num_classes = len(self.labels)

        print(f"Dataset loaded with {len(self.data)} valid images and {self.num_classes} classes")

    def _has_image(self, name):
        img_name = f"{name.replace(' ', '_')}.jpg"
        img_exists = os.path.exists(os.path.join(self.img_dir, img_name))
        return img_exists

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        name = row['Name']
        label = self.labels[name]  # Zorgt ervoor dat labels opnieuw worden toegewezen

        # Controleer of het label geldig is
        if (label >= self.num_classes):
            raise ValueError(f"Invalid label {label} for name {name}")

        img_name = f"{name.replace(' ', '_')}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if (self.transform):
            image = self.transform(image)
        return image, label, img_name

# Model
class CustomInceptionResnetV1(InceptionResnetV1):
    def __init__(self, *args, **kwargs):
        super(CustomInceptionResnetV1, self).__init__(*args, **kwargs)
        self.feature_dim = 512
        self.feature_layer = nn.Linear(self.last_linear.in_features, self.feature_dim)

    def forward(self, x, return_embeddings=False):
        # Extract features before the final layer
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        features = self.feature_layer(x)  # Reduce feature dimension to 512

        if (return_embeddings):
            return features
        else:
            return self.classifier(features)

def build_model(num_classes):
    model = CustomInceptionResnetV1(pretrained='vggface2', classify=False).to(DEVICE)
    model.num_classes = num_classes  # Set the number of classes
    model.classifier = nn.Linear(model.feature_dim, num_classes).to(DEVICE)
    model.last_bn = nn.BatchNorm1d(num_classes).to(DEVICE)

    # Zet batchnorm in evaluatiemodus
    for m in model.modules():
        if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)):
            m.eval()
            m.track_running_stats = False

    return model

# Functie om batchafbeeldingen en hun voorspellingen te visualiseren
def show_images_realtime(images, labels, predictions, img_names, epoch, fig):
    # De-normaliseer de afbeeldingen
    mean = torch.tensor([0.485, 0.456, 0.406]).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(images.device)
    images = images * std[None, :, None, None] + mean[None, :, None, None]

    # Converteer tensor naar numpy array voor weergave
    images = images.cpu().numpy()
    grid = np.transpose(images, (0, 2, 3, 1))  # HWC (Height, Width, Channels)

    # Verwijder alle bestaande subplots in de figuur
    fig.clf()

    # Maak een subplot voor elke afbeelding in de batch
    num_images = len(images)
    num_cols = 4
    num_rows = (num_images + num_cols - 1) // num_cols  # Bepaal het aantal rijen dynamisch

    fig.set_size_inches(num_cols * 3, num_rows * 3)  # Pas de figuur grootte aan

    for i in range(num_images):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.imshow(grid[i])

        true_label = labels[i].item()
        predicted_labels = predictions[:, i].cpu().numpy()

        # Bepaal of de voorspelling correct of incorrect is
        is_correct = "Correct" if (true_label in predicted_labels) else "Incorrect"

        # Zet elke voorspelling naast elkaar
        predicted_labels_str = " ".join(map(str, predicted_labels))

        ax.set_title(f"True: {true_label}\nPred: {predicted_labels_str}\n{is_correct}", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.draw()  # Update de huidige figuur
    plt.pause(0.1)  # Zorg ervoor dat de update zichtbaar is

def calculate_bonus(predictions, true_label):
    # Calculate the bonus based on the distance from the true label
    distances = torch.abs(predictions - true_label)
    max_distance = torch.max(distances).item()
    bonuses = 1 - (distances.float() / max_distance)
    return bonuses.max().item()  # Return the maximum bonus for the closest prediction

# Trainen van het model
def train_model():
    # Laad dataset
    dataset = PersonDataset(CSV_PATH, IMG_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    model = build_model(dataset.num_classes)
    model.train()

    criterion = nn.CrossEntropyLoss()  # Gebruik CrossEntropyLoss voor eenvoud
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Maak de figuur voor visualisatie
    fig = plt.figure(figsize=(16, 8))

    # Begin training
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")

        running_loss = 0.0
        correct = 0
        total = 0
        total_bonus = 0.0
        for i, (images, labels, img_names) in enumerate(dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            # Verkrijg NUM_PREDICTIONS verschillende voorspellingen voor elke afbeelding in de batch
            predictions = []
            for _ in range(NUM_PREDICTIONS):
                output = model(images)
                _, predicted = torch.max(output, 1)
                predictions.append(predicted)

            # Stapel de voorspellingen
            predictions = torch.stack(predictions)

            # Calculate accuracy
            most_common_predictions, _ = torch.mode(predictions, dim=0)
            correct += (most_common_predictions == labels).sum().item()
            total += labels.size(0)
            accuracy = correct / total

            # Calculate bonus per image
            for j in range(len(labels)):
                total_bonus += calculate_bonus(predictions[:, j], labels[j])

            # Log batch progress
            if (i % 10 == 0):  # Log every 10 batches
                print(f"Batch [{i}/{len(dataloader)}], Loss: {loss.item()}, Accuracy: {accuracy:.4f}, Bonus: {total_bonus / total:.4f}")

            # Visualiseer de afbeeldingen en de voorspellingen
            if (i % 5 == 0):  # Update de plot om de 5e batch
                show_images_realtime(images, labels, predictions, img_names, epoch, fig)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {running_loss / len(dataloader)}, Accuracy: {accuracy:.4f}, Bonus: {total_bonus / total:.4f}")

    # Bewaar het model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == '__main__':
    train_model()