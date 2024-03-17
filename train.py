import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb

# Import your dataset and model
from datasets.gym_dataset import GymOccupancyDataset
from model.time_series_transformer import TimeSeriesTransformer

# Initialize a new W&B run
wandb.init(project="gym_occupancy_prediction", entity="gujjar19", config={
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32,
    "sequence_length": 96,
    "forecast_horizon": 8,
    "d_model": 64,
    "num_layers": 3,
    "nhead": 4,
    "dim_feedforward": 256,
})

config = wandb.config

# Load dataset
dataset = GymOccupancyDataset(csv_file='data/processed_gym_occupancy_data.csv',
                              sequence_length=config.sequence_length,
                              forecast_horizon=config.forecast_horizon)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders using config
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# Define the model using config dynamically
num_features = dataset.features.shape[1]  # Adjust if necessary based on your dataset
model = TimeSeriesTransformer(num_features=num_features, config=config)


# Loss function and optimizer using config
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Device configuration (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

global_step = 0  # For more granular logging within W&B

# Training and validation loop with logging
for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        wandb.log({"batch_train_loss": loss.item()}, step=global_step)
        global_step += 1
        
    wandb.log({"epoch_train_loss": running_loss / len(train_loader), "epoch": epoch})

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            predictions = model(features)
            loss = criterion(predictions, targets)
            val_loss += loss.item()
    
    wandb.log({"epoch_val_loss": val_loss / len(val_loader), "epoch": epoch})

# Close the W&B run
wandb.finish()
