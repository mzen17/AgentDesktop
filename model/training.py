import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model.imageshot import ImageShotModel

class CursorDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        img_name = os.path.join(self.root_dir, row['filename'])
        image = Image.open(img_name).convert('RGB')
        
        # Normalize targets to [-1, 1] roughly
        # Canvas is 600x350
        w, h = 600.0, 350.0
        dx = float(row['dx']) / w
        dy = float(row['dy']) / h
        targets = torch.tensor([dx, dy], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, targets

def train():
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-4  # Reduced LR
    num_epochs = 50       # Increased epochs
    
    # Data Setup
    transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),
    ])
    
    dataset = CursorDataset(
        csv_file='model/data/labels.csv',
        root_dir='model/data/images',
        transform=transform
    )
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ImageShotModel(output_dim=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
        
        val_loss /= len(val_dataset)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    # Save Model
    os.makedirs("model/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "model/checkpoints/imageshot_model.pth")
    print("Model saved to model/checkpoints/imageshot_model.pth")

if __name__ == "__main__":
    train()
