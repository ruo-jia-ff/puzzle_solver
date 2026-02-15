import torch
from torch.utils.data import Dataset
import random
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from puzzle_piece import Piece_of_Puzzle
from generate_puzzle_pieces import Image_Puzzle
from torchvision import transforms
import cv2

EDGE_WIDTH = 16  

class EdgeDataset(Dataset):
    """
    Builds edge-pair training samples from puzzle images.
    Positive samples - True neighboring tile edges.
    Negative samples - Random non-neighbor edges, Hard negatives
    This forces the model to learn fine-grained boundary continuity instead of 
    matching only global color.
    """
    def __init__(self, image_paths, grid_size=6):
        self.samples = []
        self.edge_width = EDGE_WIDTH

        for path in image_paths:
            puzzle = Image_Puzzle(path)
            pieces = puzzle.split_into_grid(grid_size, randomize=False)

            piece_objs = []
            piece_grid = {}
            for idx, img in enumerate(pieces):
                row = idx // grid_size
                col = idx % grid_size
                p = Piece_of_Puzzle(img, row=row, col=col)
                piece_objs.append(p)
                piece_grid[(row, col)] = p

            positives = []
            negatives = []

            # Generate positive samples from true neighbors
            for p in piece_objs:
                if (p.row, p.col + 1) in piece_grid:
                    q = piece_grid[(p.row, p.col + 1)]
                    positives.append((p, "right", q, "left", 1))
                if (p.row + 1, p.col) in piece_grid:
                    q = piece_grid[(p.row + 1, p.col)]
                    positives.append((p, "bottom", q, "top", 1))

            # Generate hard negatives
            for p in piece_objs:
                for q in piece_objs:
                    if p.id == q.id:
                        continue
                    
                    manhattan_dist = abs(p.row - q.row) + abs(p.col - q.col)
                    is_neighbor = manhattan_dist == 1
                    
                    # Hard negatives: close but not adjacent
                    if manhattan_dist == 2 and not is_neighbor:
                        for s1 in ["top", "right", "bottom", "left"]:
                            for s2 in ["top", "right", "bottom", "left"]:
                                negatives.append((p, s1, q, s2, 0))
            
            # random negatives for diversity
            all_edges = [(p, side) for p in piece_objs for side in ["top", "right", "bottom", "left"]]
            for _ in range(len(positives)):
                p1, side1 = random.choice(all_edges)
                p2, side2 = random.choice(all_edges)
                
                if p1.id != p2.id:
                    manhattan_dist = abs(p1.row - p2.row) + abs(p1.col - p2.col)
                    if manhattan_dist > 2:
                        negatives.append((p1, side1, p2, side2, 0))

            self.samples += positives + negatives

        random.shuffle(self.samples)
    
    def compute_gradients(self, img_tensor):
        """Compute Sobel gradients with normalization"""
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        grad_x = (grad_x - grad_x.mean()) / (grad_x.std() + 1e-8)
        grad_y = (grad_y - grad_y.mean()) / (grad_y.std() + 1e-8)
        
        return torch.tensor(grad_x).unsqueeze(0).float(), torch.tensor(grad_y).unsqueeze(0).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p1, side1, p2, side2, label = self.samples[idx]

        e1 = p1.edges[side1]
        e2 = p2.edges[side2]

        # Get multiple strips and randomly sample one
        strips1 = p1.get_edge_strips(e1, self.edge_width)
        strips2 = p2.get_edge_strips(e2, self.edge_width)
        
        strip1 = random.choice(strips1) if strips1 else p1.get_edge_strip(e1, self.edge_width)
        strip2 = random.choice(strips2) if strips2 else p2.get_edge_strip(e2, self.edge_width)

        resize = transforms.Resize((self.edge_width, p1.size))
        strip1 = resize(strip1)
        strip2 = resize(strip2)
        
        # Data augmentation
        if random.random() < 0.5:
            augment = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(p=0.5)
            ])
            strip1 = augment(strip1)
            strip2 = augment(strip2)

        strip1 = torch.tensor(np.array(strip1)).permute(2,0,1)/255.
        strip2 = torch.tensor(np.array(strip2)).permute(2,0,1)/255.
        
        # Add gradient features
        grad1_x, grad1_y = self.compute_gradients(strip1)
        grad2_x, grad2_y = self.compute_gradients(strip2)
        
        # Concatenate: RGB (3) + gradients (2) = 5 channels per strip
        strip1_aug = torch.cat([strip1, grad1_x, grad1_y], dim=0)
        strip2_aug = torch.cat([strip2, grad2_x, grad2_y], dim=0)

        x = torch.cat([strip1_aug, strip2_aug], dim=0)  # 10 channels total
        y = torch.tensor(label).float()

        return x, y
