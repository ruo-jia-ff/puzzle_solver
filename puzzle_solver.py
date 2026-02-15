"""
Better Solver (Designed Around Your Current Trained Model)

This version does not change the model at all.
It improves the reconstruction logic instead.

Key idea:
Your model might be fine.
The greedy placement strategy might be the real issue.

So here we:
- Keep preprocessing identical to training
- Keep edge extraction consistent
- Improve neighbor scoring
- Improve placement decisions
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

from generate_puzzle_pieces import Image_Puzzle
from puzzle_piece import Piece_of_Puzzle, reset_id_generators


# ------------------------------------------------------------
# Load your existing Siamese model
# ------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "train_module",
    "./train_edge_matcher.py"
)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
SiameseNetwork = train_module.SiameseNetwork


# ------------------------------------------------------------
# Solver
# ------------------------------------------------------------

class BetterSolverForCurrentModel:
    """
    Improved reconstruction logic that works with your already-trained model.

    The model stays untouched.
    We only change how pieces are placed.
    """

    def __init__(self, model_path, edge_width=5, device="cuda"):

        self.edge_width = edge_width
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        print("Loading trained model...")

        self.model = SiameseNetwork(embedding_dim=256).to(self.device)
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )
        )
        self.model.eval()

        print(f"Model loaded on {self.device}")

        # Must match training exactly
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # --------------------------------------------------------
    # Edge extraction (must match training)
    # --------------------------------------------------------

    def get_edge_strip(self, piece, side):
        """
        No rotation here.
        Must behave exactly like training.
        """

        w, h = piece.image.size
        n = self.edge_width

        if side == "top":
            box = (0, 0, w, n)
        elif side == "bottom":
            box = (0, h - n, w, h)
        elif side == "left":
            box = (0, 0, n, h)
        else:  # right
            box = (w - n, 0, w, h)

        return piece.image.crop(box)

    def compute_edge_similarity(self, e1, e2):
        with torch.no_grad():
            t1 = self.transform(e1).unsqueeze(0).to(self.device)
            t2 = self.transform(e2).unsqueeze(0).to(self.device)
            return self.model(t1, t2).item()

    # --------------------------------------------------------
    # Rotation handling
    # --------------------------------------------------------

    def rotate_piece(self, piece, degrees):

        if degrees == 0:
            return piece

        new_img = piece.image.rotate(-degrees, expand=True)

        order = ["top", "right", "bottom", "left"]
        steps = (degrees // 90) % 4

        new_edges = {
            order[(i + steps) % 4]: piece.edges[side]
            for i, side in enumerate(order)
        }

        return Piece_of_Puzzle(new_img, id=piece.id, edges=new_edges)

    # --------------------------------------------------------
    # Neighbor scoring
    # --------------------------------------------------------

    def get_neighbor_scores(self, piece, board, row, col):

        scores = {}

        # Left neighbor
        if col > 0 and board[row][col - 1] is not None:
            left_piece = board[row][col - 1]
            e1 = self.get_edge_strip(left_piece, "right")
            e2 = self.get_edge_strip(piece, "left")
            scores["left"] = self.compute_edge_similarity(e1, e2)

        # Top neighbor
        if row > 0 and board[row - 1][col] is not None:
            top_piece = board[row - 1][col]
            e1 = self.get_edge_strip(top_piece, "bottom")
            e2 = self.get_edge_strip(piece, "top")
            scores["top"] = self.compute_edge_similarity(e1, e2)

        return scores

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    def solve_puzzle(self, pieces, grid_size, visualize=True):

        print(f"\nSolving {grid_size}x{grid_size} puzzle...")
        print("-" * 60)

        board = [[None]*grid_size for _ in range(grid_size)]

        # Fix first piece at top-left
        board[0][0] = pieces[0]
        remaining = [p for p in pieces if p.id != pieces[0].id]

        print(f"Corner fixed at (0,0) — ID {pieces[0].id}")

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
                        scores = self.get_neighbor_scores(rotated, board, r, c)

                        if scores:
                            avg = np.mean(list(scores.values()))
                            min_score = min(scores.values())
                            combined = 0.7 * avg + 0.3 * min_score
                        else:
                            combined = 0

                        if combined > best_score:
                            best_score = combined
                            best_piece = candidate
                            best_rot = rot

                if best_piece is not None:
                    board[r][c] = self.rotate_piece(best_piece, best_rot)
                    remaining = [
                        p for p in remaining
                        if p.id != best_piece.id
                    ]

                    print(f"Placed ID {best_piece.id} at ({r},{c}) "
                          f"rot={best_rot} score={best_score:.3f}")

        print("-" * 60)

        if visualize:
            self.visualize_solution(board, grid_size)

        return board

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    def visualize_solution(self, board, grid_size):

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 14))

        if grid_size == 1:
            axes = np.array([[axes]])

        for r in range(grid_size):
            for c in range(grid_size):
                ax = axes[r][c]
                piece = board[r][c]

                if piece:
                    ax.imshow(piece.image)
                    ax.set_title(f"ID:{piece.id}")

                ax.axis("off")

        plt.tight_layout()
        plt.savefig("reconstructed_puzzle_better.png", dpi=120)
        plt.show()

        print("Saved reconstructed_puzzle_better.png")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    def evaluate_reconstruction(self, board, original_puzzle, grid_size):

        ground_truth = {
            (i // grid_size, i % grid_size): i
            for i in range(len(original_puzzle.pieces))
        }

        correct = 0
        total = grid_size * grid_size

        for r in range(grid_size):
            for c in range(grid_size):
                piece = board[r][c]
                if piece and piece.id == ground_truth[(r, c)]:
                    correct += 1

        return correct / total


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    reset_id_generators()

    image_path = "./City_Scape.jpg"
    # image_path = "./dock.jpg"
    # image_path = "./Forrest.jpg"
    grid_size = 4

    print(f"Creating {grid_size}x{grid_size} puzzle from {image_path}")

    puzzle = Image_Puzzle(image_path, max_dim=1080)
    pieces_img = puzzle.split_into_grid(
        grid_size,
        randomize=True,
        keep_top_left_corner=True
    )

    pieces = [
        Piece_of_Puzzle(img, id=i)
        for i, img in enumerate(pieces_img)
    ]

    solver = BetterSolverForCurrentModel(
        model_path="./model/best_edge_matcher_v1.pth",
        edge_width=10
    )

    board = solver.solve_puzzle(pieces, grid_size)

    accuracy = solver.evaluate_reconstruction(board, puzzle, grid_size)

    print("\n" + "-" * 60)
    print(f"Accuracy: {accuracy*100:.1f}%")
    print("-" * 60)


if __name__ == "__main__":
    main()
