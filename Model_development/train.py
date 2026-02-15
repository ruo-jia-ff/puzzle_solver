import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from .dataset import EdgePairDataset
from .model import SiameseCompatibility


def train(image_paths, epochs=40, batch_size=32, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EdgePairDataset(image_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SiameseCompatibility().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Cosine Embedding Loss
    criterion = torch.nn.CosineEmbeddingLoss(margin=0.5)

    print("\nDataset Statistics:")
    print(f"Total samples: {len(dataset)}")

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for img1, img2, label in tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False
        ):

            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            # Get embeddings
            e1 = model.encoder(img1)
            e2 = model.encoder(img2)

            # Convert labels:
            # 1 -> +1 (positive)
            # 0 -> -1 (negative)
            target = label.clone()
            target[target == 0] = -1

            loss = criterion(e1, e2, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "Cosine_Embedding_Loss_model.pth")
            print("Saved best model.")

    print("Training complete.")
