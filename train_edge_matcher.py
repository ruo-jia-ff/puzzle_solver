"""
Training Script (Resume + Dataset Caching)

This version improves a few practical things:

1. Edge extraction stays consistent (no rotation tricks)
2. Hard negative sampling (nearby pieces)
3. Data augmentation
4. Checkpoint saving + resume support
5. Dataset caching to disk (so you don’t regenerate data every run)

The goal is simple:
Make training stable, reproducible, and not painfully slow.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
import os
import pickle
import hashlib

from generate_puzzle_pieces import Image_Puzzle
from puzzle_piece import Piece_of_Puzzle, reset_id_generators


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class EdgePairDataset(Dataset):
    """
    Generates edge pairs for Siamese training.

    Supports:
    - Hard negative sampling
    - Data augmentation
    - Disk caching
    """

    def __init__(
        self,
        image_paths,
        grid_sizes=[3, 4, 5, 6, 7, 8],
        edge_width=10,
        puzzles_per_image=25,
        hard_negative_ratio=0.7,
        augment=True,
        cache_dir="./dataset_cache"
    ):

        self.edge_width = edge_width
        self.hard_negative_ratio = hard_negative_ratio
        self.augment = augment
        self.pairs = []
        self.labels = []

        os.makedirs(cache_dir, exist_ok=True)

        cache_key = self._generate_cache_key(
            image_paths,
            grid_sizes,
            edge_width,
            puzzles_per_image,
            hard_negative_ratio
        )

        cache_file = os.path.join(cache_dir, f"dataset_{cache_key}.pkl")

        # Try loading from cache
        if os.path.exists(cache_file):
            print(f"Loading dataset from cache: {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                    self.pairs = cached["pairs"]
                    self.labels = cached["labels"]

                print(f"Loaded {len(self.pairs)} samples from cache")
                return
            except Exception as e:
                print(f"Cache loading failed: {e}")
                print("Regenerating dataset...")

        print("Generating dataset... (will cache after this run)")

        for img_path in image_paths:
            for grid_size in grid_sizes:
                for _ in tqdm(
                    range(puzzles_per_image),
                    desc=f"{os.path.basename(img_path)} {grid_size}x{grid_size}",
                    leave=False
                ):
                    self._generate_pairs_from_puzzle(img_path, grid_size)

        print(f"Total pairs: {len(self.pairs)}")

        # Save to cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "pairs": self.pairs,
                    "labels": self.labels
                }, f)
            print("Dataset cached successfully")
        except Exception as e:
            print(f"Failed to cache dataset: {e}")

    def _generate_cache_key(
        self,
        image_paths,
        grid_sizes,
        edge_width,
        puzzles_per_image,
        hard_negative_ratio
    ):
        config_str = (
            f"{sorted(image_paths)}"
            f"{sorted(grid_sizes)}"
            f"{edge_width}"
            f"{puzzles_per_image}"
            f"{hard_negative_ratio}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def get_transform(self):

        if self.augment:
            return transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(3)],
                    p=0.5
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def _get_edge_strip(self, piece, side):

        w, h = piece.image.size
        n = self.edge_width

        if side == "top":
            box = (0, 0, w, n)
        elif side == "bottom":
            box = (0, h - n, w, h)
        elif side == "left":
            box = (0, 0, n, h)
        else:
            box = (w - n, 0, w, h)

        return piece.image.crop(box)

    def _generate_pairs_from_puzzle(self, image_path, grid_size):

        reset_id_generators()

        puzzle = Image_Puzzle(image_path, max_dim=1080)
        pieces_img = puzzle.split_into_grid(grid_size, randomize=False)

        pieces = []
        for i, img in enumerate(pieces_img):
            row = i // grid_size
            col = i % grid_size
            pieces.append(Piece_of_Puzzle(img, row=row, col=col))

        positive_pairs = []

        # Horizontal adjacencies
        for r in range(grid_size):
            for c in range(grid_size - 1):
                left = pieces[r * grid_size + c]
                right = pieces[r * grid_size + c + 1]

                positive_pairs.append((
                    self._get_edge_strip(left, "right"),
                    self._get_edge_strip(right, "left")
                ))

        # Vertical adjacencies
        for r in range(grid_size - 1):
            for c in range(grid_size):
                top = pieces[r * grid_size + c]
                bottom = pieces[(r + 1) * grid_size + c]

                positive_pairs.append((
                    self._get_edge_strip(top, "bottom"),
                    self._get_edge_strip(bottom, "top")
                ))

        negative_pairs = []
        num_negatives = len(positive_pairs)
        num_hard = int(num_negatives * self.hard_negative_ratio)

        indexed = [(p, i // grid_size, i % grid_size)
                   for i, p in enumerate(pieces)]

        # Hard negatives (nearby but wrong)
        for _ in range(num_hard):
            p1, r1, c1 = random.choice(indexed)
            nearby = [
                (p, r, c)
                for p, r, c in indexed
                if abs(r - r1) <= 2
                and abs(c - c1) <= 2
                and (r, c) != (r1, c1)
            ]

            if nearby:
                p2, _, _ = random.choice(nearby)
                s1 = random.choice(["top", "bottom", "left", "right"])
                s2 = random.choice(["top", "bottom", "left", "right"])
                negative_pairs.append((
                    self._get_edge_strip(p1, s1),
                    self._get_edge_strip(p2, s2)
                ))

        # Easy negatives
        all_edges = []
        for p in pieces:
            for side in ["top", "bottom", "left", "right"]:
                all_edges.append(self._get_edge_strip(p, side))

        for _ in range(num_negatives - num_hard):
            e1, e2 = random.sample(all_edges, 2)
            negative_pairs.append((e1, e2))

        for e1, e2 in positive_pairs:
            self.pairs.append((e1, e2))
            self.labels.append(1)

        for e1, e2 in negative_pairs:
            self.pairs.append((e1, e2))
            self.labels.append(0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        e1, e2 = self.pairs[idx]
        label = self.labels[idx]

        transform = self.get_transform()

        return (
            transform(e1),
            transform(e2),
            torch.tensor(label, dtype=torch.float32)
        )


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class SiameseNetwork(nn.Module):

    def __init__(self, embedding_dim=256):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.similarity = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        f = self.feature_extractor(x)
        f = f.view(f.size(0), -1)
        return self.fc(f)

    def forward(self, e1, e2):
        emb1 = self.forward_once(e1)
        emb2 = self.forward_once(e2)
        combined = torch.cat([emb1, emb2], dim=1)
        return self.similarity(combined)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def save_checkpoint(epoch, model, optimizer, scheduler,
                    best_val_acc, patience_counter, path):

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "patience_counter": patience_counter
    }, path)

    print(f"Checkpoint saved: {path}")


def train_model(model, train_loader, val_loader,
                num_epochs=150, device="cuda",
                resume_from=None):

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_val_acc = 0
    patience = 15
    patience_counter = 0
    start_epoch = 0

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        patience_counter = ckpt.get("patience_counter", 0)

    for epoch in range(start_epoch, num_epochs):

        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for e1, e2, labels in pbar:

            e1 = e1.to(device)
            e2 = e2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(e1, e2).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for e1, e2, labels in val_loader:
                e1 = e1.to(device)
                e2 = e2.to(device)
                labels = labels.to(device)

                outputs = model(e1, e2).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(f"\nEpoch {epoch+1}")
        print(f"Train Acc: {train_acc:.4f}")
        print(f"Val Acc:   {val_acc:.4f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./model/best_edge_matcher.pth")
            save_checkpoint(
                epoch, model, optimizer, scheduler,
                best_val_acc, patience_counter,
                "./model/checkpoint_best.pth"
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    image_paths = [
        "./City_Scape.jpg",
        "./dock.jpg",
        "./Forrest.jpg"
    ]

    print("-" * 60)
    print("Training edge matcher with dataset caching")
    print("-" * 60)

    dataset = EdgePairDataset(
        image_paths=image_paths,
        cache_dir="./dataset_cache"
    )

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SiameseNetwork().to(device)

    train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        resume_from=None
    )

    print("Training finished.")


if __name__ == "__main__":
    main()
