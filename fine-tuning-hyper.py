import torch
from pathlib import Path

# Set up paths and directories
repo = Path('yolov5')
weights = repo / 'yolov5s.pt'
data = repo / 'data' / 'my_custom_dataset.yaml'
hyp = repo / 'data' / 'hyp.finetune.yaml'

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=weights)

# Fine-tune model on custom dataset with custom hyperparameters
model.finetune(data=data, epochs=10, hyp=hyp)