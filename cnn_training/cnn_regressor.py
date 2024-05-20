import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

class LocalizationCNN(nn.Module):
    def __init__(self):
        # super(LocalizationCNN, self).__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(15, 32, kernel_size=5, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.BatchNorm2d(32),

        #     nn.Conv2d(32, 64, kernel_size=10, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.BatchNorm2d(64),

        #     nn.Conv2d(64, 128, kernel_size=5, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.BatchNorm2d(128)
        # )
        
        # self.regressor = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(128 * 1 * 1, 256),  # Adjust the size based on the output of the last conv layer
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 6)  # Output layer for x, y coordinates and velocities
        # )
        
        super(LocalizationCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=10, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),  # Adjust the size based on the output of the last conv layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 6)  # Output layer for x, y coordinates and velocities
        )
        

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

# Load and prepare the dataset
def load_data(X, Y, training_size=0.8):
    dataset = TensorDataset(X, Y)

    # Splitting dataset into training and validation
    train_size = int(training_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset

def train_model(model, device, train_loader, val_loader, optimizer, loss_function, epochs):
    model.train()
    training_losses = []
    validate_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        training_avg_loss = total_loss / len(train_loader.dataset)
        training_losses.append(training_avg_loss) # average loss for each epoch
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            total_loss += loss.item() * data.size(0)
        validate_avg_loss = total_loss / len(val_loader.dataset)
        validate_losses.append(validate_avg_loss)
        print(f'Epoch {epoch+1}, Training Loss: {training_avg_loss:.7f}, Validation Loss: {validate_avg_loss:.7f}')
    return training_losses, validate_losses

def main():
    X = torch.load('training_images.pt')
    Y = torch.load('position_labels.pt')
    # Y = torch.load('velocity_labels.pt')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_dataset, val_dataset = load_data(X, Y)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # data set sorted into batches of 64 and shuffled
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize the model and other components
    model = LocalizationCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    # Training and Validation
    training_losses, validate_losses = train_model(model, device, train_loader, val_loader, optimizer, loss_function, epochs=500)
    
    return training_losses, validate_losses
    

if __name__ == 'main':
    training_losses, validate_losses = main()