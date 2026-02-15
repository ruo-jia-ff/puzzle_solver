from PIL import Image
import math
import matplotlib.pyplot as plt
import random

class Image_Puzzle:

    def __init__(self, image_path, max_dim=1080):
        self.image = self._crop_to_square_center(image_path, max_dim)
        self.pieces = []
        self.placement_dict = {}

    def _crop_to_square_center(self, image_path, max_dim):
        image = Image.open(image_path)
        width, height = image.size
        side_length = min(width, height)

        left = (width - side_length) // 2
        top = (height - side_length) // 2
        right = left + side_length
        bottom = top + side_length
        square = image.crop((left, top, right, bottom))

        if max_dim is not None and side_length > max_dim:
            square = square.resize((max_dim, max_dim), Image.LANCZOS)
        return square

    def split_into_grid(self, n, randomize=True, keep_top_left_corner=True):
        width, height = self.image.size
        piece_width = width // n
        piece_height = height // n

        pieces = []
        placement_dict = {}

        for row in range(n):
            for col in range(n):
                left = col * piece_width
                upper = row * piece_height
                right = (col + 1) * piece_width
                lower = (row + 1) * piece_height
                piece = self.image.crop((left, upper, right, lower))
                pieces.append(piece)
                idx = row * n + col
                placement_dict[idx] = {'shuffle_id': idx, 'angle': 0}

        if randomize:
            new_pieces = [None] * len(pieces)
            indices = list(range(len(pieces)))
            if keep_top_left_corner:
                indices = indices[1:]
                random.shuffle(indices)
                indices = [0] + indices
            else:
                random.shuffle(indices)

            for new_idx, orig_idx in enumerate(indices):
                angle = random.choice([0, 90, 180, 270]) if orig_idx != 0 else 0
                rotated = pieces[orig_idx].rotate(angle, expand=True)
                new_pieces[new_idx] = rotated
                placement_dict[new_idx]["shuffle_id"] = orig_idx
                placement_dict[new_idx]["angle"] = angle
            pieces = new_pieces

        self.pieces = pieces
        self.placement_dict = placement_dict
        return pieces

    def show_all_pieces(self, dimensions=None, show_labels=False):
        """
        Displays all pieces in an n x n grid.
        Optionally shows shuffled index and rotation.
        """
        pieces = self.pieces
        if not pieces:
            print("No pieces to display.")
            return None

        if dimensions:
            h, w = dimensions
        else:
            h = w = int(math.sqrt(len(pieces)))

        _, axes = plt.subplots(h, w, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(pieces[i])
            ax.axis('off')

            if show_labels and self.placement_dict:
                shuffle_id = self.placement_dict[i]['shuffle_id']
                angle = self.placement_dict[i]['angle']
                # index in center
                ax.text(pieces[i].size[0] / 2, pieces[i].size[1] / 2, f"{i}",
                        color='white', ha='center', va='center', fontsize=12, weight='bold')

        plt.tight_layout()
        plt.show()

    # def _show_reassembled(self, pieces, show_labels=False):
    #     h = w = int(math.sqrt(len(pieces)))
    #     _, axes = plt.subplots(h, w, figsize=(8, 8))
    #     for i, ax in enumerate(axes.flat):
    #         ax.imshow(pieces[i])
    #         ax.axis("off")

    #         if show_labels and self.placement_dict:
    #             shuffle_id = self.placement_dict[i]['shuffle_id']
    #             angle = self.placement_dict[i]['angle']
    #             ax.text(pieces[i].size[0] / 2, pieces[i].size[1] / 2, f"{shuffle_id}",
    #                     color='white', ha='center', va='center', fontsize=12, weight='bold')
    #             ax.text(pieces[i].size[0] / 2, pieces[i].size[1] / 2 + 25, f"{angle}°",
    #                     color='yellow', ha='center', va='center', fontsize=10)

        # plt.tight_layout()
        # plt.show()

    def _rotate_back(self, piece, angle):
        if angle == 0:
            return piece

        inv_angle = (360 - angle) % 360
        rotated = piece.rotate(inv_angle, expand=True)

        w, h = piece.size
        target = h if angle in (90, 270) else w

        rw, rh = rotated.size
        left = (rw - target) // 2
        top = (rh - target) // 2

        return rotated.crop((left, top, left + target, top + target))

    def _show_reassembled(self, reassembled, info, show_labels):
        n = int(math.sqrt(len(reassembled)))
        _, axes = plt.subplots(n, n, figsize=(8, 8))

        for i, ax in enumerate(axes.flat):
            piece = reassembled[i]
            if piece:
                ax.imshow(piece)

            ax.axis("off")

            if show_labels and i in info:
                size = piece.size[0]
                ax.text(size / 2, size / 2, f"{info[i]['from_position']}",
                        color="white", ha="center", va="center",
                        fontsize=12, weight="bold")
                ax.text(size / 2, size / 2 + 25, f"{info[i]['angle']}°",
                        color="yellow", ha="center", va="center",
                        fontsize=10)

        plt.tight_layout()
        plt.show()


    def reassemble(self, show=True, show_labels=False):
        """
        Reassemble pieces using placement_dict.
        Shows original image with labels indicating where each piece came from.
        """
        reassembled = [None] * len(self.pieces)
        reassemble_info = {}

        for current_idx, info in self.placement_dict.items():
            shuffle_id = info["shuffle_id"]
            angle = info["angle"]

            piece = self.pieces[current_idx]
            original_piece = self._rotate_back(piece, angle)

            reassembled[shuffle_id] = original_piece
            reassemble_info[shuffle_id] = {
                "from_position": current_idx,
                "angle": angle
            }

        if show:
            self._show_reassembled(reassembled, reassemble_info, show_labels)

        return reassembled


    def get_pieces(self):
        return self.pieces