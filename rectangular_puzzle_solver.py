"""
Rectangular Puzzle Solver (No Cropping)

This version stops forcing images to be square.

Instead of center-cropping (which throws away useful information),
we keep the full image and split it into a rectangular grid
that matches the original aspect ratio.

Why this matters:
- Cropping removes distinctive features.
- Square grids distort natural composition.
- Rectangular grids better reflect real images.

So this version:
- Preserves full image
- Automatically chooses a reasonable rows x cols layout
- Uses the same multi-feature edge matching as before
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.ndimage import convolve
import random

from puzzle_piece import Piece_of_Puzzle, reset_id_generators


# ------------------------------------------------------------
# Puzzle Creation (Full Image, No Cropping)
# ------------------------------------------------------------

class RectangularImagePuzzle:
    """
    Splits the full image into a rectangular grid.
    Never crops to square.
    """

    def __init__(self, image_path, max_dim=1920):

        self.image = Image.open(image_path)
        original_size = self.image.size

        width, height = self.image.size

        # Resize only if too large (keep aspect ratio)
        if max(width, height) > max_dim:
            scale = max_dim / max(width, height)
            new_size = (
                int(width * scale),
                int(height * scale)
            )
            self.image = self.image.resize(new_size, Image.LANCZOS)
            print(f"Resized image from {original_size} to {self.image.size}")
        else:
            print(f"Using full image: {self.image.size}")

        self.pieces = []
        self.placement_dict = {}

    def calculate_optimal_grid(self, target_pieces=16):
        """
        Choose rows and columns that roughly match
        the image aspect ratio.
        """

        width, height = self.image.size
        aspect_ratio = width / height

        best_rows, best_cols = 4, 4
        best_diff = float("inf")

        for rows in range(2, 8):
            cols = max(2, int(rows * aspect_ratio))
            total = rows * cols
            diff = abs(total - target_pieces)

            if diff < best_diff:
                best_diff = diff
                best_rows = rows
                best_cols = cols

        print(f"Aspect ratio: {aspect_ratio:.2f}")
        print(f"Grid selected: {best_rows} x {best_cols}")

        return best_rows, best_cols

    def split_into_grid(self, rows=None, cols=None,
                        randomize=True,
                        keep_top_left_corner=True):

        if rows is None or cols is None:
            rows, cols = self.calculate_optimal_grid()

        width, height = self.image.size
        piece_w = width // cols
        piece_h = height // rows

        print(f"Piece size: {piece_w} x {piece_h}")

        pieces = []
        placement_dict = {}

        for r in range(rows):
            for c in range(cols):

                left = c * piece_w
                top = r * piece_h

                right = (c + 1) * piece_w if c < cols - 1 else width
                bottom = (r + 1) * piece_h if r < rows - 1 else height

                piece = self.image.crop((left, top, right, bottom))
                pieces.append(piece)

                idx = r * cols + c
                placement_dict[idx] = {"shuffle_id": idx, "angle": 0}

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
                new_pieces[new_idx] = pieces[orig_idx].rotate(angle, expand=True)
                placement_dict[new_idx]["shuffle_id"] = orig_idx
                placement_dict[new_idx]["angle"] = angle

            pieces = new_pieces

        self.pieces = pieces
        self.placement_dict = placement_dict
        self.grid_rows = rows
        self.grid_cols = cols

        return pieces

    def show_all_pieces(self, show_labels=False):

        if not self.pieces:
            print("No pieces available.")
            return

        rows, cols = self.grid_rows, self.grid_cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, 10))

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(rows * cols):
            r = i // cols
            c = i % cols

            ax = axes[r][c]

            if i < len(self.pieces):
                ax.imshow(self.pieces[i])
                if show_labels:
                    ax.set_title(str(i))

            ax.axis("off")

        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------
# Edge Feature Extraction
# ------------------------------------------------------------

class EdgeFeatureExtractor:

    def __init__(self, edge_width=40):
        self.edge_width = edge_width

    def get_edge_strip(self, piece, side):

        w, h = piece.image.size
        n = min(self.edge_width, min(w, h) // 3)

        if side == "top":
            box = (0, 0, w, n)
        elif side == "bottom":
            box = (0, h - n, w, h)
        elif side == "left":
            box = (0, 0, n, h)
        else:
            box = (w - n, 0, w, h)

        return piece.image.crop(box)

    def compute_color_histogram(self, edge_img, bins=32):

        img = np.array(edge_img)

        hist_r = np.histogram(img[:, :, 0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(img[:, :, 1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(img[:, :, 2], bins=bins, range=(0, 256))[0]

        hist = np.concatenate([hist_r, hist_g, hist_b])
        return hist / (hist.sum() + 1e-6)

    def compute_gradient_features(self, edge_img):

        gray = np.array(edge_img.convert("L")).astype(float)

        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        gx = convolve(gray, sobel_x)
        gy = convolve(gray, sobel_y)

        magnitude = np.sqrt(gx**2 + gy**2)

        return {
            "magnitude_mean": np.mean(magnitude),
            "magnitude_std": np.std(magnitude)
        }

    def compute_edge_color_profile(self, edge_img):

        img = np.array(edge_img)
        h, w, _ = img.shape

        if h < w:
            return img[-1, :, :]
        else:
            return img[:, -1, :]

    def compute_texture_features(self, edge_img):

        from scipy.ndimage import uniform_filter

        gray = np.array(edge_img.convert("L")).astype(float)

        mean = uniform_filter(gray, size=3)
        mean_sq = uniform_filter(gray**2, size=3)
        variance = mean_sq - mean**2

        return {
            "texture_mean": np.mean(variance),
            "texture_std": np.std(variance)
        }


# ------------------------------------------------------------
# Rectangular Puzzle Solver
# ------------------------------------------------------------

class RectangularPuzzleSolver:

    def __init__(self, edge_width=40):
        self.feature_extractor = EdgeFeatureExtractor(edge_width)

    def compute_edge_compatibility(self, e1, e2):

        if e1.size != e2.size:
            target = (
                min(e1.size[0], e2.size[0]),
                min(e1.size[1], e2.size[1])
            )
            e1 = e1.resize(target)
            e2 = e2.resize(target)

        scores = {}

        # Color continuity
        c1 = self.feature_extractor.compute_edge_color_profile(e1)
        c2 = self.feature_extractor.compute_edge_color_profile(e2)
        color_diff = np.mean(np.abs(c1.astype(float) - c2.astype(float))) / 255.0
        scores["color"] = 1.0 / (1.0 + color_diff * 5)

        # Histogram similarity
        h1 = self.feature_extractor.compute_color_histogram(e1)
        h2 = self.feature_extractor.compute_color_histogram(e2)
        hist_dist = distance.correlation(h1, h2)
        scores["hist"] = 1.0 / (1.0 + hist_dist)

        # Gradient similarity
        g1 = self.feature_extractor.compute_gradient_features(e1)
        g2 = self.feature_extractor.compute_gradient_features(e2)
        mag_diff = abs(g1["magnitude_mean"] - g2["magnitude_mean"])
        scores["grad"] = 1.0 / (1.0 + mag_diff / 50.0)

        # Texture similarity
        t1 = self.feature_extractor.compute_texture_features(e1)
        t2 = self.feature_extractor.compute_texture_features(e2)
        tex_diff = abs(t1["texture_mean"] - t2["texture_mean"])
        scores["texture"] = 1.0 / (1.0 + tex_diff / 100.0)

        weights = {
            "color": 0.40,
            "hist": 0.20,
            "grad": 0.25,
            "texture": 0.15
        }

        return sum(scores[k] * weights[k] for k in scores)

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

    def solve_puzzle(self, pieces, rows, cols, visualize=True):

        print(f"\nSolving {rows}x{cols} rectangular puzzle")
        print("-" * 60)

        board = [[None]*cols for _ in range(rows)]
        board[0][0] = pieces[0]

        remaining = [p for p in pieces if p.id != pieces[0].id]

        for r in range(rows):
            for c in range(cols):

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
                            left = self.feature_extractor.get_edge_strip(board[r][c-1], "right")
                            right = self.feature_extractor.get_edge_strip(rotated, "left")
                            scores.append(self.compute_edge_compatibility(left, right))

                        if r > 0 and board[r-1][c]:
                            top = self.feature_extractor.get_edge_strip(board[r-1][c], "bottom")
                            bottom = self.feature_extractor.get_edge_strip(rotated, "top")
                            scores.append(self.compute_edge_compatibility(top, bottom))

                        if scores:
                            avg = np.mean(scores)
                            min_score = min(scores)
                            score = 0.7 * avg + 0.3 * min_score
                        else:
                            score = 0

                        if score > best_score:
                            best_score = score
                            best_piece = candidate
                            best_rot = rot

                if best_piece:
                    board[r][c] = self.rotate_piece(best_piece, best_rot)
                    remaining = [p for p in remaining if p.id != best_piece.id]

                    print(f"Placed ID {best_piece.id} at ({r},{c}) "
                          f"rot={best_rot} score={best_score:.3f}")

        print("-" * 60)

        if visualize:
            self.visualize_solution(board, rows, cols)

        return board

    def visualize_solution(self, board, rows, cols):

        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])

        for r in range(rows):
            for c in range(cols):
                ax = axes[r][c]
                piece = board[r][c]

                if piece:
                    ax.imshow(piece.image)
                    ax.set_title(f"ID:{piece.id}")

                ax.axis("off")

        plt.tight_layout()
        plt.savefig("reconstructed_puzzle_rectangular.png", dpi=120)
        plt.show()

        print("Saved reconstructed_puzzle_rectangular.png")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    test_images = [
        "./City_Scape.jpg",
        "./dock.jpg",
        "./Forrest.jpg"
    ]

    solver = RectangularPuzzleSolver(edge_width=40)

    for path in test_images:

        print("\n" + "="*60)
        print(f"Testing: {path}")
        print("="*60)

        reset_id_generators()

        puzzle = RectangularImagePuzzle(path, max_dim=1920)
        pieces_img = puzzle.split_into_grid(
            randomize=True,
            keep_top_left_corner=True
        )

        rows, cols = puzzle.grid_rows, puzzle.grid_cols

        pieces = [
            Piece_of_Puzzle(img, id=i)
            for i, img in enumerate(pieces_img)
        ]

        board = solver.solve_puzzle(pieces, rows, cols)

    print("\nDone.")


if __name__ == "__main__":
    main()
