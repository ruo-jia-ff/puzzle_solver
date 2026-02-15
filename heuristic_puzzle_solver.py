"""
Heuristic Puzzle Solver (Multi-Feature Version)

This one doesn't use ML.
No training.
No neural nets.

Just classic computer vision tricks combined together:

1. Color continuity along edges
2. Color histogram similarity
3. Gradient similarity (Sobel-based)
4. Simple texture estimation (local variance)

Each metric gets a weight.
We combine them into a final score.

It’s greedy placement (row by row),
but the matching itself is fairly robust.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.ndimage import convolve
import random
import sys


from generate_puzzle_pieces import Image_Puzzle
from puzzle_piece import Piece_of_Puzzle, reset_id_generators


# ------------------------------------------------------------
# Edge Feature Extraction
# ------------------------------------------------------------

class EdgeFeatureExtractor:
    """
    Extracts multiple types of features from a puzzle edge.
    """

    def __init__(self, edge_width=40):
        self.edge_width = edge_width

    def get_edge_strip(self, piece, side):
        """Return a cropped strip from one side of the piece."""
        w, h = piece.image.size
        n = self.edge_width

        if side == "top":
            box = (0, 0, w, n)
        elif side == "bottom":
            box = (0, h - n, w, h)
        elif side == "left":
            box = (0, 0, n, h)
        elif side == "right":
            box = (w - n, 0, w, h)
        else:
            raise ValueError("Invalid side")

        return piece.image.crop(box)

    def compute_color_histogram(self, edge_img, bins=32):
        """Compute normalized RGB histogram."""
        img = np.array(edge_img)

        hist_r = np.histogram(img[:, :, 0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(img[:, :, 1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(img[:, :, 2], bins=bins, range=(0, 256))[0]

        hist = np.concatenate([hist_r, hist_g, hist_b])
        hist = hist / (hist.sum() + 1e-6)

        return hist

    def compute_gradient_features(self, edge_img):
        """Compute simple Sobel-based gradient stats."""
        gray = np.array(edge_img.convert("L")).astype(float)

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0,  0,  0],
                            [1,  2,  1]])

        grad_x = convolve(gray, sobel_x)
        grad_y = convolve(gray, sobel_y)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return {
            "magnitude_mean": np.mean(magnitude),
            "magnitude_std": np.std(magnitude),
        }

    def compute_texture_features(self, edge_img):
        """Use local variance as a cheap texture estimate."""
        from scipy.ndimage import uniform_filter

        gray = np.array(edge_img.convert("L")).astype(float)

        mean = uniform_filter(gray, size=3)
        mean_sq = uniform_filter(gray**2, size=3)
        variance = mean_sq - mean**2

        return {
            "texture_mean": np.mean(variance),
            "texture_std": np.std(variance),
        }

    def compute_boundary_pixels(self, edge_img):
        """Return the actual pixels along the touching boundary."""
        img = np.array(edge_img)
        h, w, _ = img.shape

        # Decide which row/column represents the boundary
        if h < w:
            return img[-1, :, :]  # bottom row
        else:
            return img[:, -1, :]  # right column


# ------------------------------------------------------------
# Solver
# ------------------------------------------------------------

class HeuristicPuzzleSolver:
    """
    Greedy solver that uses multi-feature edge scoring.
    """

    def __init__(self, edge_width=40):
        self.feature_extractor = EdgeFeatureExtractor(edge_width)

    def compute_edge_compatibility(self, edge1, edge2):
        """
        Compare two edge strips and return compatibility score (0–1).
        Higher = better match.
        """

        if edge1.size != edge2.size:
            target = (
                min(edge1.size[0], edge2.size[0]),
                min(edge1.size[1], edge2.size[1])
            )
            edge1 = edge1.resize(target)
            edge2 = edge2.resize(target)

        scores = {}

        # ---- 1. Color continuity (40%) ----
        b1 = self.feature_extractor.compute_boundary_pixels(edge1)
        b2 = self.feature_extractor.compute_boundary_pixels(edge2)

        color_diff = np.mean(np.abs(b1.astype(float) - b2.astype(float))) / 255.0
        scores["color"] = 1.0 / (1.0 + color_diff * 5)

        # ---- 2. Histogram similarity (20%) ----
        h1 = self.feature_extractor.compute_color_histogram(edge1)
        h2 = self.feature_extractor.compute_color_histogram(edge2)

        hist_dist = distance.correlation(h1, h2)
        scores["hist"] = 1.0 / (1.0 + hist_dist)

        # ---- 3. Gradient similarity (25%) ----
        g1 = self.feature_extractor.compute_gradient_features(edge1)
        g2 = self.feature_extractor.compute_gradient_features(edge2)

        mag_diff = abs(g1["magnitude_mean"] - g2["magnitude_mean"])
        scores["grad"] = 1.0 / (1.0 + mag_diff / 50.0)

        # ---- 4. Texture similarity (15%) ----
        t1 = self.feature_extractor.compute_texture_features(edge1)
        t2 = self.feature_extractor.compute_texture_features(edge2)

        tex_diff = abs(t1["texture_mean"] - t2["texture_mean"])
        scores["texture"] = 1.0 / (1.0 + tex_diff / 100.0)

        weights = {
            "color": 0.40,
            "hist": 0.20,
            "grad": 0.25,
            "texture": 0.15
        }

        final_score = sum(scores[k] * weights[k] for k in scores)

        return final_score

    def rotate_piece(self, piece, degrees):
        """Rotate piece by 0, 90, 180, 270."""
        if degrees == 0:
            return piece

        new_img = piece.image.rotate(-degrees, expand=True)
        return Piece_of_Puzzle(new_img, id=piece.id)

    def solve_puzzle(self, pieces, grid_size, visualize=True):

        print("\n" + "="*60)
        print(f"Solving {grid_size}x{grid_size} puzzle (heuristic mode)")
        print("="*60)
        print("Using color + histogram + gradient + texture\n")

        board = [[None for _ in range(grid_size)]
                 for _ in range(grid_size)]

        # Fix first piece in top-left
        board[0][0] = pieces[0]
        remaining = [p for p in pieces if p.id != pieces[0].id]

        print(f"Placed corner piece ID {pieces[0].id} at (0,0)")

        for row in range(grid_size):
            for col in range(grid_size):

                if board[row][col] is not None:
                    continue

                best_score = -1
                best_piece = None
                best_rot = 0

                for candidate in remaining:
                    for rot in [0, 90, 180, 270]:

                        rotated = self.rotate_piece(candidate, rot)
                        scores = []

                        # Check left neighbor
                        if col > 0 and board[row][col-1] is not None:
                            left_piece = board[row][col-1]

                            e1 = self.feature_extractor.get_edge_strip(left_piece, "right")
                            e2 = self.feature_extractor.get_edge_strip(rotated, "left")

                            scores.append(self.compute_edge_compatibility(e1, e2))

                        # Check top neighbor
                        if row > 0 and board[row-1][col] is not None:
                            top_piece = board[row-1][col]

                            e1 = self.feature_extractor.get_edge_strip(top_piece, "bottom")
                            e2 = self.feature_extractor.get_edge_strip(rotated, "top")

                            scores.append(self.compute_edge_compatibility(e1, e2))

                        if scores:
                            score = 0.7 * np.mean(scores) + 0.3 * min(scores)
                        else:
                            score = 0

                        if score > best_score:
                            best_score = score
                            best_piece = candidate
                            best_rot = rot

                if best_piece:
                    board[row][col] = self.rotate_piece(best_piece, best_rot)
                    remaining = [p for p in remaining if p.id != best_piece.id]

                    print(f"Placed ID {best_piece.id} at ({row},{col}) "
                          f"rot={best_rot} score={best_score:.3f}")

        print("="*60 + "\n")

        if visualize:
            self.visualize_solution(board, grid_size)

        return board

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
        plt.savefig("reconstructed_puzzle_heuristic.png", dpi=120)
        plt.show()

        print("Saved: reconstructed_puzzle_heuristic.png")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    test_images = [
        ("./City_Scape.jpg", "City"),
        ("./dock.jpg", "Dock"),
        ("./Forrest.jpg", "Forest"),
    ]

    grid_size = 3

    print("\n" + "="*70)
    print("Heuristic Puzzle Solver")
    print("="*70)
    print("No training. Pure CV.\n")

    solver = HeuristicPuzzleSolver(edge_width=40)

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

        board = solver.solve_puzzle(pieces, grid_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
