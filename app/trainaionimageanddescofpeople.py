import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import glob
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Global variables for configuration
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
NUM_EPOCHS = 20
MODEL_SAVE_DIR = 'saved_ai'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'person_recognition_model.pth')

# Ensure the save directory exists
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# Data preprocessing with simpler augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PersonDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.valid_data = []
        for _, row in self.data.iterrows():
            name = row['Name'].replace(' ', '_')
            img_path = os.path.join(img_dir, f"{name}.jpg")
            if os.path.exists(img_path):
                self.valid_data.append((row['Name'], img_path))
        self.name_to_idx = {name: idx for idx, (name, _) in enumerate(self.valid_data)}

    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        name, img_path = self.valid_data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.name_to_idx[name]
        return image, label

    def get_num_classes(self):
        return len(self.name_to_idx)

def evaluate_model(dataloader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    avg_loss = running_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PersonDataset(
        csv_file='datasets/dataset.csv',
        img_dir='input_images',
        transform=transform
    )
    train_data, val_data = train_test_split(dataset.valid_data, test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, [dataset.valid_data.index(entry) for entry in train_data])
    val_dataset = torch.utils.data.Subset(dataset, [dataset.valid_data.index(entry) for entry in val_data])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, dataset.get_num_classes())
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions
        val_loss, val_accuracy = evaluate_model(val_loader, model, criterion, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        scheduler.step()
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_model()