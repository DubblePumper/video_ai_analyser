import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt
import numpy as np

# Configuratie
LEARNING_RATE = 0.001
BATCH_SIZE = 32
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
        
        # Print de labels om te controleren of ze goed zijn toegewezen
        print(f"Labels: {self.labels}")
        
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
        if label >= self.num_classes:
            raise ValueError(f"Invalid label {label} for name {name}")
        
        img_name = f"{name.replace(' ', '_')}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_name


# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        p_t = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return loss.mean()


# Center Loss
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim=512, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        # Haal de bijbehorende centroids op voor de labels van de batch
        centers_batch = self.centers.to(features.device).index_select(0, labels.long())
        
        # Bereken het verlies door de batch features met de centroids te vergelijken
        loss = torch.sum((features - centers_batch) ** 2, dim=1)  # Vermijd de foute dimensie door `dim=1` toe te voegen
        return loss.mean()  # Gemiddelde over de batch


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

        if return_embeddings:
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
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            m.track_running_stats = False

    return model


# Functie om batchafbeeldingen te visualiseren
def show_images(images, labels, predicted, img_names, epoch, output_dir="output_images"):
    # De-normaliseer de afbeeldingen
    mean = torch.tensor([0.485, 0.456, 0.406]).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(images.device)
    images = images * std[None, :, None, None] + mean[None, :, None, None]
    
    # Converteer tensor naar numpy array voor weergave
    images = images.cpu().numpy()
    grid = np.transpose(images, (0, 2, 3, 1))  # HWC (Height, Width, Channels)
    
    # Maak de output map aan voor de huidige epoch
    epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
    os.makedirs(epoch_output_dir, exist_ok=True)
    
    # Sla de afbeelding op als een PNG bestand
    for i, img in enumerate(grid):
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img_pil)
        
        true_label = labels[i].item()
        predicted_label = predicted[i].item()
        img_name = img_names[i]
        
        # Toevoegen van de naam van de echte en voorspelde labels
        if true_label != predicted_label:
            draw.text((10, 10), f"True: {img_name} - {true_label}", fill="red")
            draw.text((10, 30), f"Pred: {predicted_label}", fill="red")
        else:
            draw.text((10, 10), f"True: {img_name} - {true_label}", fill="green")
            draw.text((10, 30), f"Pred: {predicted_label}", fill="green")
        
        # Opslaan van de afbeelding
        img_pil.save(os.path.join(epoch_output_dir, f"{img_name}.png"))
    
    print(f"Afbeeldingen voor epoch {epoch+1} opgeslagen in {epoch_output_dir}")


def train_model():
    # Laad dataset
    dataset = PersonDataset(CSV_PATH, IMG_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Bouw het model
    model = build_model(dataset.num_classes)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_focal = FocalLoss()
    center_loss = CenterLoss(num_classes=dataset.num_classes, feat_dim=512, alpha=0.5).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, img_names) in enumerate(dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Vooruitpassing
            features = model(images, return_embeddings=True)
            outputs = model.classifier(features)
            
            # Verlies berekening
            ce_loss = criterion_ce(outputs, labels)
            focal_loss = criterion_focal(outputs, labels)
            c_loss = center_loss(features, labels)
            
            # Totale verlies
            loss = ce_loss + focal_loss + c_loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}, Accuracy: {100 * correct / total}%")
                
            # Sla afbeeldingen op elke 10e batch
            if batch_idx % 10 == 0:
                show_images(images, labels, predicted, img_names, epoch)
        
        print(f"Epoch {epoch+1} completed, Loss: {running_loss/len(dataloader)}, Accuracy: {100 * correct / total}%")
        
    # Opslaan van het model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model opgeslagen op {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()