import torch
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from puzzle_board import PuzzleBoard
from generate_puzzle_pieces import Image_Puzzle
from puzzle_piece import Piece_of_Puzzle
from edge_model import EdgeNet
from torchvision import transforms
import random
from collections import defaultdict
import math
from copy import deepcopy
import cv2

""" 
Uses the trained ML model to reconstruct a shuffled image by:
1. Scoring boundary compatibility between tiles
2. Greedy placement with multi-neighbor constraints
3. Local refinement targeting weak edges
Tile 0 is permanently anchored at the top-left by default as per instruction.
"""

EDGE_WIDTH = 16 
GRID = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EdgeNet().to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

puzzle = Image_Puzzle("dock.jpg")
tiles = puzzle.split_into_grid(GRID)

pieces = []
for i, img in enumerate(tiles):
    pieces.append(Piece_of_Puzzle(img, row=i//GRID, col=i%GRID))

resize = transforms.Resize((EDGE_WIDTH, pieces[0].size))

def rotate_piece(piece, degrees):
    rotated_img = piece.image.rotate(-degrees, expand=True)
    order = ["top", "right", "bottom", "left"]
    steps = (degrees // 90) % 4
    rotated_edges = {order[(i + steps) % 4]: piece.edges[side] for i, side in enumerate(order)}
    return Piece_of_Puzzle(rotated_img, row=piece.row, col=piece.col, id=piece.id, edges=rotated_edges)

def compute_gradients(img_tensor):
    """Compute Sobel gradients with normalization"""
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_x = (grad_x - grad_x.mean()) / (grad_x.std() + 1e-8)
    grad_y = (grad_y - grad_y.mean()) / (grad_y.std() + 1e-8)
    
    return torch.tensor(grad_x).unsqueeze(0).float(), torch.tensor(grad_y).unsqueeze(0).float()

def edge_distance(p1, side1, p2, side2):
    """Compute L2 distance between edge strips"""
    e1 = p1.edges[side1]
    e2 = p2.edges[side2]
    
    strip1 = p1.get_edge_strip(e1, EDGE_WIDTH)
    strip2 = p2.get_edge_strip(e2, EDGE_WIDTH)
    
    strip1 = resize(strip1)
    strip2 = resize(strip2)
    
    arr1 = np.array(strip1).astype(float) / 255.
    arr2 = np.array(strip2).astype(float) / 255.
    
    return np.sqrt(np.mean((arr1 - arr2) ** 2))

def cnn_score(p1, side1, p2, side2):
    e1 = p1.edges[side1]
    e2 = p2.edges[side2]
    
    strip1 = p1.get_edge_strip(e1, EDGE_WIDTH)
    strip2 = p2.get_edge_strip(e2, EDGE_WIDTH)
    
    strip1 = resize(strip1)
    strip2 = resize(strip2)
    
    t1 = torch.tensor(np.array(strip1)).permute(2,0,1)/255.
    t2 = torch.tensor(np.array(strip2)).permute(2,0,1)/255.

    grad1_x, grad1_y = compute_gradients(t1)
    grad2_x, grad2_y = compute_gradients(t2)
    
    strip1_aug = torch.cat([t1, grad1_x, grad1_y], dim=0)
    strip2_aug = torch.cat([t2, grad2_x, grad2_y], dim=0)
    
    x = torch.cat([strip1_aug, strip2_aug], 0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        score = torch.sigmoid(model(x)).item()
    
    return score

def edge_cost(p1, side1, p2, side2):
    cnn = cnn_score(p1, side1, p2, side2)
    dist = edge_distance(p1, side1, p2, side2)
    return 0.8 * (1 - cnn) + 0.2 * dist  

def grid_energy(grid, grid_size):
    energy = 0
    for row in range(grid_size):
        for col in range(grid_size):
            if grid[row][col] is None:
                continue
            p = grid[row][col]
            
            if col < grid_size - 1 and grid[row][col+1] is not None:
                energy += edge_cost(p, "right", grid[row][col+1], "left")
            
            if row < grid_size - 1 and grid[row+1][col] is not None:
                energy += edge_cost(p, "bottom", grid[row+1][col], "top")
    
    return energy

print("Building compatibility matrix with rotation search...")

grid_size = GRID
grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
placed = set()
piece_rotations = {}

anchor_piece = None
for p in pieces:
    if p.id == 0:
        anchor_piece = p
        break

if anchor_piece is None:
    raise ValueError("Piece with id=0 not found!")

grid[0][0] = anchor_piece
placed.add(anchor_piece.id)
piece_rotations[anchor_piece.id] = 0

print(f"Anchored piece {anchor_piece.id} at (0,0) with rotation=0")

print("Greedy placement with multi-neighbor constraint...")

for row in range(grid_size):
    for col in range(grid_size):
        if grid[row][col] is not None:
            continue
        
        best_piece = None
        best_rotation = 0
        best_cost = float('inf')
        
        for p in pieces:
            if p.id in placed:
                continue
            
            for rot in [0, 90, 180, 270]:
                rotated = rotate_piece(p, rot)
                total_cost = 0
                neighbor_count = 0
                
                if col > 0 and grid[row][col-1] is not None:
                    left = grid[row][col-1]
                    total_cost += edge_cost(left, "right", rotated, "left")
                    neighbor_count += 1
                
                if row > 0 and grid[row-1][col] is not None:
                    top = grid[row-1][col]
                    total_cost += edge_cost(top, "bottom", rotated, "top")
                    neighbor_count += 1
                
                if row > 0 and col > 0 and neighbor_count < 2:
                    continue 
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_piece = rotated
                    best_rotation = rot
        
        if best_piece:
            grid[row][col] = best_piece
            placed.add(best_piece.id)
            piece_rotations[best_piece.id] = best_rotation
        else:
            for p in pieces:
                if p.id not in placed:
                    grid[row][col] = p
                    placed.add(p.id)
                    piece_rotations[p.id] = 0
                    break

print(f"Initial placement complete. Energy: {grid_energy(grid, grid_size):.4f}")
print("\nIdentifying weak edges...")
weak_edge_tiles = set()
THRESHOLD = 0.5

for row in range(grid_size):
    for col in range(grid_size):
        p = grid[row][col]
        
        if col < grid_size - 1:
            cost = edge_cost(p, "right", grid[row][col+1], "left")
            if cost > THRESHOLD:
                weak_edge_tiles.add((row, col))
                weak_edge_tiles.add((row, col+1))
        
        if row < grid_size - 1:
            cost = edge_cost(p, "bottom", grid[row+1][col], "top")
            if cost > THRESHOLD:
                weak_edge_tiles.add((row, col))
                weak_edge_tiles.add((row+1, col))

print(f"Found {len(weak_edge_tiles)} tiles touching weak edges")

print("Targeted refinement pass...")

for iteration in range(5):
    improved = False
    
    for (row, col) in weak_edge_tiles:
        if row == 0 and col == 0:  
            continue
        
        current_energy = grid_energy(grid, grid_size)
        current_piece = grid[row][col]
        
        for rot in [90, 180, 270]:
            rotated = rotate_piece(current_piece, rot)
            grid[row][col] = rotated
            new_energy = grid_energy(grid, grid_size)
            
            if new_energy < current_energy:
                current_energy = new_energy
                current_piece = rotated
                improved = True
            else:
                grid[row][col] = current_piece
        
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if nr == 0 and nc == 0:  
                    continue
                
                grid[row][col], grid[nr][nc] = grid[nr][nc], grid[row][col]
                new_energy = grid_energy(grid, grid_size)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                else:
                    grid[row][col], grid[nr][nc] = grid[nr][nc], grid[row][col]
    
    print(f"Iteration {iteration+1}: Energy = {grid_energy(grid, grid_size):.4f}")
    if not improved:
        break

print(f"Final energy: {grid_energy(grid, grid_size):.4f}")


print(f"\nVerifying anchor: grid[0][0].id = {grid[0][0].id} (expected: 0)")
assert grid[0][0].id == 0, "ERROR: Tile 0 not at (0,0)!"


print("\nValidating reconstruction quality...")
weak_edges = []

for row in range(grid_size):
    for col in range(grid_size):
        p = grid[row][col]
        
        if col < grid_size - 1:
            cost = edge_cost(p, "right", grid[row][col+1], "left")
            if cost > THRESHOLD:
                weak_edges.append((row, col, "right", cost))
                print(f"Weak match at ({row},{col}) right - cost: {cost:.3f}")
        
        if row < grid_size - 1:
            cost = edge_cost(p, "bottom", grid[row+1][col], "top")
            if cost > THRESHOLD:
                weak_edges.append((row, col, "bottom", cost))
                print(f"Weak match at ({row},{col}) bottom - cost: {cost:.3f}")

print(f"\nTotal weak edges: {len(weak_edges)}")
if len(weak_edges) == 0:
    print("✓ All edges have strong matches!")
else:
    print(f"⚠ {len(weak_edges)} edges need improvement")


board = PuzzleBoard()
for p in pieces:
    board.add_piece(p)

for row in range(grid_size):
    for col in range(grid_size):
        if grid[row][col] is not None:
            piece = grid[row][col]
            board.placed_pieces[piece.id] = ((col, row), piece)
            board.occupied_positions[(col, row)] = piece.id

board.visualize_chains()

"""
In Future:
- we can replace greedy placement with Hungarian / ILP
- Add contrastive loss to the model
- Extend solver to arbitrary grid sizes
"""