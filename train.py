import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb

# Import your dataset and model
from datasets.gym_dataset import GymOccupancyDataset
from model.time_series_transformer import TimeSeriesTransformer

# Initialize a new W&B run
wandb.init(project="gym_occupancy_prediction", entity="gujjar19")

# Define your configuration
config = wandb.config
config.learning_rate = 0.001
config.epochs = 20
config.batch_size = 32
config.sequence_length = 96
config.forecast_horizon = 8
config.d_model = 64
config.num_layers = 3
config.nhead = 4
config.dim_feedforward = 256

# Load dataset
dataset = GymOccupancyDataset(csv_file='path_to_your_processed_csv.csv', sequence_length=config.sequence_length, forecast_horizon=config.forecast_horizon)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders using config
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# Define the model using config
num_features = dataset.features.shape[1]  # Adjust if necessary based on your dataset
model = TimeSeriesTransformer(num_features=num_features, num_layers=config.num_layers, d_model=config.d_model, nhead=config.nhead, dim_feedforward=config.dim_feedforward)

# Loss function and optimizer using config
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Device configuration (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training and validation loop with logging
for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    # Log training metrics
    wandb.log({"train_loss": running_loss / len(train_loader)})

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            predictions = model(features)
            loss = criterion(predictions, targets.unsqueeze(-1))
            val_loss += loss.item()
    
    # Log validation metrics
    wandb.log({"val_loss": val_loss / len(val_loader)})

# Close the W&B run
wandb.finish()
