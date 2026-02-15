"""
Graph-Based Puzzle Solver (Global Optimization Version)

So earlier versions placed puzzle pieces greedily —
pick the best-looking match at each step and hope it works out.

This version does something smarter:
it builds a full cost matrix and solves everything at once
using the Hungarian algorithm.

Instead of just matching edges,
we look at the whole piece:
- overall color
- spatial layout
- brightness
- rough edge summaries

It's still heuristic-based,
but now the placement is globally optimal.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import random

from puzzle_piece import Piece_of_Puzzle, reset_id_generators


# ------------------------------------------------------------
# Puzzle Builder
# ------------------------------------------------------------

class RectangularImagePuzzle:
    """
    Splits an image into a rectangular grid.

    Keeps the full image (no cropping),
    preserves aspect ratio,
    and optionally scrambles everything.
    """

    def __init__(self, image_path, max_dim=1920):
        self.image = Image.open(image_path)

        # Resize if too large (but keep aspect ratio)
        width, height = self.image.size
        if max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            self.image = self.image.resize(
                (int(width * scale), int(height * scale)),
                Image.LANCZOS
            )

        self.pieces = []
        self.placement_dict = {}

    def calculate_optimal_grid(self, target_pieces=12):
        """
        Try to find a grid size that roughly matches
        the image aspect ratio.

        Not mathematically perfect.
        Just a practical guess.
        """
        width, height = self.image.size
        aspect_ratio = width / height

        best_rows, best_cols = 3, 3
        best_diff = float("inf")

        for rows in range(2, 6):
            cols = int(rows * aspect_ratio)
            cols = max(cols, 2)

            total = rows * cols
            diff = abs(total - target_pieces)

            if diff < best_diff:
                best_diff = diff
                best_rows = rows
                best_cols = cols

        return best_rows, best_cols

    def split_into_grid(self, rows=None, cols=None,
                        randomize=True,
                        keep_top_left_corner=True):
        """
        Split image into rows x cols pieces.

        If randomize=True:
            - shuffle pieces
            - rotate randomly
            - optionally keep top-left fixed
        """

        if rows is None or cols is None:
            rows, cols = self.calculate_optimal_grid()

        width, height = self.image.size
        piece_w = width // cols
        piece_h = height // rows

        pieces = []
        placement_dict = {}

        # Crop pieces
        for r in range(rows):
            for c in range(cols):
                left = c * piece_w
                upper = r * piece_h
                right = (c + 1) * piece_w if c < cols - 1 else width
                lower = (r + 1) * piece_h if r < rows - 1 else height

                piece = self.image.crop((left, upper, right, lower))
                pieces.append(piece)

                idx = r * cols + c
                placement_dict[idx] = {"shuffle_id": idx, "angle": 0}

        # Shuffle and rotate
        if randomize:
            indices = list(range(len(pieces)))

            if keep_top_left_corner:
                indices = indices[1:]
                random.shuffle(indices)
                indices = [0] + indices
            else:
                random.shuffle(indices)

            new_pieces = [None] * len(pieces)

            for new_idx, orig_idx in enumerate(indices):
                angle = random.choice([0, 90, 180, 270]) if orig_idx != 0 else 0
                rotated = pieces[orig_idx].rotate(angle, expand=True)

                new_pieces[new_idx] = rotated
                placement_dict[new_idx]["shuffle_id"] = orig_idx
                placement_dict[new_idx]["angle"] = angle

            pieces = new_pieces

        self.pieces = pieces
        self.placement_dict = placement_dict
        self.grid_rows = rows
        self.grid_cols = cols

        return pieces


# ------------------------------------------------------------
# Feature Extraction
# ------------------------------------------------------------

class GlobalFeatureExtractor:
    """
    Extracts simple whole-piece features.

    Nothing fancy:
    - mean RGB
    - std RGB
    - small color histogram
    - 3x3 spatial averages
    - edge color summaries
    """

    @staticmethod
    def extract_color_features(piece_img):
        img = np.array(piece_img)

        mean = np.mean(img, axis=(0, 1))
        std = np.std(img, axis=(0, 1))

        hist_r = np.histogram(img[:, :, 0], bins=16, range=(0, 256))[0]
        hist_g = np.histogram(img[:, :, 1], bins=16, range=(0, 256))[0]
        hist_b = np.histogram(img[:, :, 2], bins=16, range=(0, 256))[0]

        hist = np.concatenate([hist_r, hist_g, hist_b])
        hist = hist / (hist.sum() + 1e-6)

        return {"mean": mean, "std": std, "hist": hist}

    @staticmethod
    def extract_spatial_features(piece_img):
        img = np.array(piece_img)
        h, w = img.shape[:2]

        cell_h = h // 3
        cell_w = w // 3

        features = []
        for i in range(3):
            for j in range(3):
                cell = img[i*cell_h:(i+1)*cell_h,
                           j*cell_w:(j+1)*cell_w]
                features.extend(np.mean(cell, axis=(0, 1)))

        return np.array(features)

    @staticmethod
    def extract_edge_features(piece_img, edge_width=20):
        w, h = piece_img.size
        n = min(edge_width, min(w, h) // 4)

        img = np.array(piece_img)

        top = np.mean(img[:n, :, :], axis=(0, 1))
        bottom = np.mean(img[h-n:h, :, :], axis=(0, 1))
        left = np.mean(img[:, :n, :], axis=(0, 1))
        right = np.mean(img[:, w-n:w, :], axis=(0, 1))

        return {"top": top, "bottom": bottom,
                "left": left, "right": right}


# ------------------------------------------------------------
# Solver
# ------------------------------------------------------------

class GraphBasedPuzzleSolver:

    def __init__(self):
        self.feature_extractor = GlobalFeatureExtractor()

    def compute_compatibility_matrix(self, pieces, rows, cols):
        num_pieces = len(pieces)
        num_positions = rows * cols

        print(f"Computing cost matrix ({num_pieces} x {num_positions})...")

        features = []
        for p in pieces:
            features.append({
                "color": self.feature_extractor.extract_color_features(p.image),
                "spatial": self.feature_extractor.extract_spatial_features(p.image),
                "edges": self.feature_extractor.extract_edge_features(p.image)
            })

        cost_matrix = np.zeros((num_pieces, num_positions))

        for i, feat in enumerate(features):
            for pos in range(num_positions):
                r = pos // cols
                c = pos % cols
                cost_matrix[i, pos] = self.compute_position_cost(
                    feat, r, c, rows, cols
                )

        return cost_matrix

    def compute_position_cost(self, feat, row, col, rows, cols):
        cost = 0.0

        mean_color = feat["color"]["mean"]
        spatial = feat["spatial"]
        edges = feat["edges"]

        brightness = np.mean(mean_color)

        # Simple assumptions
        if row == 0:
            cost += (255 - brightness) / 255.0
        elif row == rows - 1:
            cost += brightness / 255.0

        if col == 0:
            cost += 0.1 * np.std(edges["left"]) / 128.0
        elif col == cols - 1:
            cost += 0.1 * np.std(edges["right"]) / 128.0

        top_b = np.mean(spatial[:9])
        bottom_b = np.mean(spatial[18:27])

        if top_b < bottom_b:
            cost += 0.2

        return cost

    def solve_with_assignment(self, pieces, rows, cols):
        print("\n" + "="*60)
        print(f"Solving {rows}x{cols} puzzle (global mode)")
        print("="*60)

        cost_matrix = self.compute_compatibility_matrix(pieces, rows, cols)

        print("Running Hungarian algorithm...")
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        print("Done. Optimal assignment found.\n")

        board = [[None for _ in range(cols)] for _ in range(rows)]

        for piece_idx, pos_idx in zip(row_ind, col_ind):
            r = pos_idx // cols
            c = pos_idx % cols
            board[r][c] = pieces[piece_idx]

            print(f"Placed piece {pieces[piece_idx].id} "
                  f"at ({r},{c}) | "
                  f"cost={cost_matrix[piece_idx,pos_idx]:.3f}")

        print("="*60 + "\n")
        return board

    def visualize_solution(self, board, rows, cols):
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for r in range(rows):
            for c in range(cols):
                ax = axes[r, c]
                piece = board[r][c]

                if piece is not None:
                    ax.imshow(piece.image)
                    ax.text(
                        piece.image.size[0] / 2,
                        piece.image.size[1] / 2,
                        f"ID:{piece.id}",
                        color="white",
                        ha="center",
                        va="center",
                        fontsize=10,
                        weight="bold",
                        bbox=dict(boxstyle="round",
                                  facecolor="purple",
                                  alpha=0.7)
                    )

                ax.axis("off")

        plt.tight_layout()
        plt.savefig("reconstructed_puzzle_graph.png", dpi=120)
        plt.show()

        print("Saved: reconstructed_puzzle_graph.png")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    test_images = [
        ("./City_Scape.jpg", "City"),
        ("./dock.jpg", "Dock"),
        ("./Forrest.jpg", "Forest")
    ]

    print("\n" + "="*70)
    print("Graph-Based Puzzle Solver")
    print("="*70)
    print("Global assignment instead of greedy placement.\n")

    solver = GraphBasedPuzzleSolver()
    results = []

    for path, name in test_images:

        print("\n" + "="*70)
        print(f"Testing: {name}")
        print("="*70 + "\n")

        reset_id_generators()

        puzzle = RectangularImagePuzzle(path)
        pieces_img = puzzle.split_into_grid(
            rows=3,
            cols=4,
            randomize=True,
            keep_top_left_corner=True
        )

        rows, cols = puzzle.grid_rows, puzzle.grid_cols
        pieces = [Piece_of_Puzzle(img, id=i)
                  for i, img in enumerate(pieces_img)]

        board = solver.solve_with_assignment(pieces, rows, cols)
        solver.visualize_solution(board, rows, cols)

    print("\nDone.")


if __name__ == "__main__":
    main()
