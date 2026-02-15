import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from torch.utils.data import DataLoader
from edge_dataset import EdgeDataset
from edge_model import EdgeNet

"""
To train and learn boundary compatibility between puzzle tiles.
The model is trained using only the three provided images as per 
the task info, by generating positive and negative edge pairs. 
Early stopping is used to prevent overfitting given the limited dataset size.
"""

images = ["dock.jpg", "Forrest.jpg", "City_Scape.jpg"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training...")
dataset = EdgeDataset(images)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = EdgeNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()

print(f"Training on {device} with {len(dataset)} samples")

best_loss = float('inf')
patience = 20
patience_counter = 0

for epoch in range(150):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/150 - Loss: {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), "model.pt")
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"\nBest loss: {best_loss:.4f}")
print("Model saved to model.pt")

"""
Future improvements:
1. Add explicit validation split when more images are available
2. Use mixed precision for faster GPU training
3. Replace BCE with contrastive loss for stronger metric learning
"""