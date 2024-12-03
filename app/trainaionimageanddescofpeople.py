import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PersonDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        print(f"Loading dataset from {csv_file}")
        self.data = pd.read_csv(csv_file)
        print(f"Found {len(self.data)} entries in CSV")
        
        self.img_dir = img_dir
        print(f"Looking for images in {img_dir}")
        
        self.transform = transform
        self.valid_data = []
        
        # Filter only entries with existing images
        for idx, row in self.data.iterrows():
            name = row['Name'].replace(' ', '_')
            img_path = os.path.join(img_dir, f"{name}.png")
            if os.path.exists(img_path):
                self.valid_data.append((row['Name'], img_path))
            else:
                print(f"Image not found for {name} at {img_path}")
        
        print(f"Found {len(self.valid_data)} valid images")
        
        if len(self.valid_data) == 0:
            raise ValueError("No valid images found! Please check your image directory and file names.")
            
        # Create name to index mapping
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
        return len(self.valid_data)

def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create dataset and dataloader
        dataset = PersonDataset(
            csv_file='app/datasets/dataset.csv',  # Remove 'app/' prefix
            img_dir='app/input_images',           # Remove 'app/' prefix
            transform=transform
        )
        
        if dataset.get_num_classes() == 0:
            raise ValueError("Dataset is empty!")
            
        print(f"Dataset created successfully with {dataset.get_num_classes()} classes")
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Load pre-trained ResNet model
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, dataset.get_num_classes())
        model = model.to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            epoch_loss = running_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        # Save the model
        torch.save(model.state_dict(), 'person_recognition_model.pth')
        print("Training complete! Model saved as 'person_recognition_model.pth'")
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
