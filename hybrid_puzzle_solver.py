"""
Hybrid Puzzle Solver (ML + Heuristic Fallback)

This version combines both worlds:

- If the image has strong variation / edges → use the trained Siamese model.
- If the image is too smooth or uniform → fall back to gradient matching.

Idea:
ML works great on structured, high-detail images.
Heuristics work better when everything looks the same (sky, water, forest fog).

So instead of picking one approach —
we decide dynamically.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import importlib.util


from generate_puzzle_pieces import Image_Puzzle
from puzzle_piece import Piece_of_Puzzle, reset_id_generators


# ------------------------------------------------------------
# Load Siamese Model
# ------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "train_module",
    "./train_edge_matcher.py"
)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
SiameseNetwork = train_module.SiameseNetwork


# ------------------------------------------------------------
# Image Analysis (Decide Which Solver to Use)
# ------------------------------------------------------------

class ImageAnalyzer:
    """
    Looks at the original image and decides:
    ML or heuristic?
    """

    @staticmethod
    def compute_variance(image):
        gray = np.array(image.convert("L"))
        return np.var(gray)

    @staticmethod
    def compute_edge_strength(image):
        gray = np.array(image.convert("L"))

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0,  0,  0],
                            [1,  2,  1]])

        h, w = gray.shape
        gx = np.zeros((h-2, w-2))
        gy = np.zeros((h-2, w-2))

        for i in range(1, h-1):
            for j in range(1, w-1):
                region = gray[i-1:i+2, j-1:j+2]
                gx[i-1, j-1] = np.sum(region * sobel_x)
                gy[i-1, j-1] = np.sum(region * sobel_y)

        return np.mean(np.sqrt(gx**2 + gy**2))

    @staticmethod
    def should_use_ml(image):

        variance = ImageAnalyzer.compute_variance(image)
        edge_strength = ImageAnalyzer.compute_edge_strength(image)

        variance_threshold = 800
        edge_threshold = 15

        use_ml = (
            variance > variance_threshold
            or edge_strength > edge_threshold
        )

        print("\nImage analysis:")
        print(f"  Variance: {variance:.1f}")
        print(f"  Edge strength: {edge_strength:.1f}")
        print(f"  → Decision: {'Use ML model' if use_ml else 'Use heuristic'}")

        return use_ml


# ------------------------------------------------------------
# ML Solver
# ------------------------------------------------------------

class MLSolver:
    """
    Uses trained Siamese network for edge similarity.
    """

    def __init__(self, model_path, edge_width=10, device="cuda"):
        self.edge_width = edge_width
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        print("Loading ML model...")

        self.model = SiameseNetwork(embedding_dim=256).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path,
                       map_location=self.device,
                       weights_only=True)
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("ML model ready.")

    def get_edge_strip(self, piece, side):
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

    def compute_similarity(self, e1, e2):
        with torch.no_grad():
            t1 = self.transform(e1).unsqueeze(0).to(self.device)
            t2 = self.transform(e2).unsqueeze(0).to(self.device)
            return self.model(t1, t2).item()

    def rotate_piece(self, piece, degrees):
        if degrees == 0:
            return piece
        new_img = piece.image.rotate(-degrees, expand=True)
        return Piece_of_Puzzle(new_img, id=piece.id)

    def solve(self, pieces, grid_size):

        print("Using ML-based matching...")

        board = [[None]*grid_size for _ in range(grid_size)]
        board[0][0] = pieces[0]

        remaining = [p for p in pieces if p.id != pieces[0].id]

        for r in range(grid_size):
            for c in range(grid_size):

                if board[r][c] is not None:
                    continue

                best_score = -1
                best_piece = None
                best_rot = 0

                for candidate in remaining:
                    for rot in [0, 90, 180, 270]:

                        rotated = self.rotate_piece(candidate, rot)
                        scores = []

                        if c > 0 and board[r][c-1]:
                            left = self.get_edge_strip(board[r][c-1], "right")
                            right = self.get_edge_strip(rotated, "left")
                            scores.append(self.compute_similarity(left, right))

                        if r > 0 and board[r-1][c]:
                            top = self.get_edge_strip(board[r-1][c], "bottom")
                            bottom = self.get_edge_strip(rotated, "top")
                            scores.append(self.compute_similarity(top, bottom))

                        score = np.mean(scores) if scores else 0

                        if score > best_score:
                            best_score = score
                            best_piece = candidate
                            best_rot = rot

                if best_piece:
                    board[r][c] = self.rotate_piece(best_piece, best_rot)
                    remaining = [p for p in remaining if p.id != best_piece.id]

        return board


# ------------------------------------------------------------
# Heuristic Solver (Gradient-Based)
# ------------------------------------------------------------

class HeuristicSolver:

    def __init__(self, edge_width=30):
        self.edge_width = edge_width

    def get_edge_strip(self, piece, side):
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

    def gradient_similarity(self, e1, e2):

        a = np.array(e1)
        b = np.array(e2)

        if a.shape != b.shape:
            return 0.0

        grad1 = np.mean(np.abs(np.diff(a, axis=0)))
        grad2 = np.mean(np.abs(np.diff(b, axis=0)))

        diff = abs(grad1 - grad2)
        return 1.0 / (1.0 + diff)

    def rotate_piece(self, piece, degrees):
        if degrees == 0:
            return piece
        new_img = piece.image.rotate(-degrees, expand=True)
        return Piece_of_Puzzle(new_img, id=piece.id)

    def solve(self, pieces, grid_size):

        print("Using gradient-based matching...")

        board = [[None]*grid_size for _ in range(grid_size)]
        board[0][0] = pieces[0]

        remaining = [p for p in pieces if p.id != pieces[0].id]

        for r in range(grid_size):
            for c in range(grid_size):

                if board[r][c]:
                    continue

                best_score = -1
                best_piece = None
                best_rot = 0

                for candidate in remaining:
                    for rot in [0, 90, 180, 270]:

                        rotated = self.rotate_piece(candidate, rot)
                        scores = []

                        if c > 0 and board[r][c-1]:
                            left = self.get_edge_strip(board[r][c-1], "right")
                            right = self.get_edge_strip(rotated, "left")
                            scores.append(self.gradient_similarity(left, right))

                        if r > 0 and board[r-1][c]:
                            top = self.get_edge_strip(board[r-1][c], "bottom")
                            bottom = self.get_edge_strip(rotated, "top")
                            scores.append(self.gradient_similarity(top, bottom))

                        score = np.mean(scores) if scores else 0

                        if score > best_score:
                            best_score = score
                            best_piece = candidate
                            best_rot = rot

                if best_piece:
                    board[r][c] = self.rotate_piece(best_piece, best_rot)
                    remaining = [p for p in remaining if p.id != best_piece.id]

        return board


# ------------------------------------------------------------
# Hybrid Controller
# ------------------------------------------------------------

class HybridPuzzleSolver:

    def __init__(self, model_path, edge_width=10, device="cuda"):
        print("Initializing hybrid solver...")
        self.ml_solver = MLSolver(model_path, edge_width, device)
        self.heuristic_solver = HeuristicSolver(edge_width=30)
        self.analyzer = ImageAnalyzer()
        print("Hybrid solver ready.\n")

    def solve_puzzle(self, pieces, grid_size, original_image):

        print("\n" + "="*60)
        print(f"Solving {grid_size}x{grid_size} puzzle (hybrid mode)")
        print("="*60)

        use_ml = self.analyzer.should_use_ml(original_image)

        if use_ml:
            board = self.ml_solver.solve(pieces, grid_size)
            method = "ML"
        else:
            board = self.heuristic_solver.solve(pieces, grid_size)
            method = "Heuristic"

        print("="*60 + "\n")

        return board, method


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    test_images = [
        ("./City_Scape.jpg", "City"),
        ("./dock.jpg", "Dock"),
        ("./Forrest.jpg", "Forest"),
    ]

    grid_size = 4

    solver = HybridPuzzleSolver(
        model_path="./model/best_edge_matcher_v1.pth",
        edge_width=10
    )

    results = []

    for path, name in test_images:

        print("\n" + "="*70)
        print(f"Testing: {name}")
        print("="*70)

        reset_id_generators()

        puzzle = Image_Puzzle(path, max_dim=1080)
        pieces_img = puzzle.split_into_grid(
            grid_size,
            randomize=True,
            keep_top_left_corner=True
        )

        pieces = [Piece_of_Puzzle(img, id=i)
                  for i, img in enumerate(pieces_img)]

        board, method = solver.solve_puzzle(
            pieces,
            grid_size,
            puzzle.image
        )

        results.append((name, method))

    print("\nDone.")
    print("\nSummary:")
    for name, method in results:
        print(f"{name:10} → {method}")


if __name__ == "__main__":
    main()
