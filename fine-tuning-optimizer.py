import torch
from pathlib import Path
from torch.optim import Adam

# Set up paths and directories
repo = Path('yolov5')
weights = repo / 'yolov5s.pt'
data = repo / 'data' / 'my_custom_dataset.yaml'

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=weights)

# Set up custom optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Fine-tune model on custom dataset with custom optimizer
model.finetune(data=data, epochs=10, optimizer=optimizer)