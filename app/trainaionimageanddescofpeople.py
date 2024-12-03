import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from facenet_pytorch import InceptionResnetV1

# Configuratie
LEARNING_RATE = 0.0001
BATCH_SIZE = 5
NUM_EPOCHS = 20
IMG_SIZE = 160  # VGGFace2 gebruikt 160x160 afbeeldingen
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paden
CSV_PATH = os.path.join(SCRIPT_DIR, 'datasets', 'dataset.csv')
IMG_DIR = os.path.join(SCRIPT_DIR, 'input_images')
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, 'saved_ai', 'person_recognition_model_vggface2.pth')
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Transformaties
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalisatie voor VGGFace2
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
        if not img_exists:
            print(f"Image not found: {img_name}")
        return img_exists
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        name = row['Name']
        label = self.labels[name]  # Zorgt ervoor dat labels opnieuw worden toegewezen
        img_name = f"{name.replace(' ', '_')}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Model
def build_model(num_classes):
    # Laad het voorgetrainde VGGFace2-model
    model = InceptionResnetV1(pretrained='vggface2', classify=False).to(DEVICE)
    
    # Vervang de classificatielaag
    num_ftrs = model.last_linear.in_features
    model.last_linear = nn.Linear(num_ftrs, num_classes).to(DEVICE)
    
    # Zet batchnorm statistieken uit om de fout te vermijden
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)
    
    return model

# Model bouwen en batchnorm uitschakelen
dataset = PersonDataset(CSV_PATH, IMG_DIR, transform=transform)
model = build_model(dataset.num_classes)  # Gebruik de instantie van dataset

# Train functie
def train_model():
    print("Starting training process...")
    dataset = PersonDataset(CSV_PATH, IMG_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = build_model(dataset.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(NUM_EPOCHS):
        model.train()  # Zorg ervoor dat het model in trainingsmodus is
        running_loss = 0.0
        correct = 0
        total = 0
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} started")
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Controleer of de labels binnen het bereik van het aantal klassen liggen
            assert labels.max() < dataset.num_classes, "Label out of bounds!"
            
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Processed: {batch_idx * len(images)}/{len(dataloader.dataset)}")
        
        epoch_loss = running_loss / len(dataloader)
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Model opslaan
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model opgeslagen: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    print("Initializing training script...")
    train_model()
    print("Training script finished")