import torch
from torch.utils.data import Dataset
from torchvision import transforms
from puzzle_piece import Piece_of_Puzzle
from generate_puzzle_pieces import Image_Puzzle
from PIL import Image
import random


class EdgePairDataset(Dataset):
    """
    Rotation-safe dataset.
    Positives are generated BEFORE rotation using edge IDs.
    Edge strips are extracted dynamically at retrieval time.
    """

    def __init__(self, image_paths, grid_sizes=[3, 4, 5, 6], strip_width=8):
        self.samples = []
        self.strip_width = strip_width

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        for path in image_paths:
            for n in grid_sizes:

       
                #   Create puzzle WITHOUT rotation
                puzzle = Image_Puzzle(path)
                pieces = puzzle.split_into_grid(n, randomize=False)

                # Wrap into Piece_of_Puzzle objects
                piece_objs = [Piece_of_Puzzle(img) for img in pieces]

                #  Generate ALL labels using EDGE IDs
                self._generate_pairs_with_edge_ids(piece_objs, n)

                # Apply random rotation AFTER label creation
                # Top-left (index 0) remains unrotated

                for idx, p in enumerate(piece_objs):
                    if idx == 0:
                        continue  
                    deg = random.choice([0, 90, 180, 270])
                    p.rotate(deg)

                # # Store rotated pieces for strip extraction
                # for sample in self.samples:
                #     sample["pieces"] = piece_objs


    # Generate pairs using EDGE IDs (rotation invariant)

    def _generate_pairs_with_edge_ids(self, pieces, n):
        num_pieces = len(pieces)

        for idx1 in range(num_pieces):
            i1, j1 = divmod(idx1, n)
            p1 = pieces[idx1]

            for idx2 in range(idx1 + 1, num_pieces):
                i2, j2 = divmod(idx2, n)
                p2 = pieces[idx2]

                if self._is_horizontal_neighbor(i1, j1, i2, j2):
                    self._store_sample(
                        idx1, idx2,
                        p1.edges["right"],
                        p2.edges["left"],
                        label=1,
                        pieces=pieces
                    )
                else:
                    self._store_sample(
                        idx1, idx2,
                        p1.edges["right"],
                        p2.edges["left"],
                        label=0,
                        pieces=pieces
                    )

                if self._is_horizontal_neighbor(i2, j2, i1, j1):
                    self._store_sample(
                        idx2, idx1,
                        p2.edges["right"],
                        p1.edges["left"],
                        label=1,
                        pieces=pieces
                    )
                else:
                    self._store_sample(
                        idx2, idx1,
                        p2.edges["right"],
                        p1.edges["left"],
                        label=0,
                        pieces=pieces
                    )

                if self._is_vertical_neighbor(i1, j1, i2, j2):
                    self._store_sample(
                        idx1, idx2,
                        p1.edges["bottom"],
                        p2.edges["top"],
                        label=1,
                        pieces=pieces
                    )
                else:
                    self._store_sample(
                        idx1, idx2,
                        p1.edges["bottom"],
                        p2.edges["top"],
                        label=0,
                        pieces=pieces
                    )

                if self._is_vertical_neighbor(i2, j2, i1, j1):
                    self._store_sample(
                        idx2, idx1,
                        p2.edges["bottom"],
                        p1.edges["top"],
                        label=1,
                        pieces=pieces
                    )
                else:
                    self._store_sample(
                        idx2, idx1,
                        p2.edges["bottom"],
                        p1.edges["top"],
                        label=0,
                        pieces=pieces
                    )


    # Neighbor checks

    def _is_horizontal_neighbor(self, i1, j1, i2, j2):
        return (i1 == i2) and (j2 == j1 + 1)

    def _is_vertical_neighbor(self, i1, j1, i2, j2):
        return (j1 == j2) and (i2 == i1 + 1)

    # Store supervision sample

    def _store_sample(self, idx1, idx2, edge_id1, edge_id2, label, pieces):
        self.samples.append({
            "idx1": idx1,
            "idx2": idx2,
            "edge_id1": edge_id1,
            "edge_id2": edge_id2,
            "label": label,
            "pieces": pieces  
        })

    # Canonical strip extraction (rotation robust)

    def _get_canonical_strip(self, piece, edge_id):
        strip = piece.get_edge_strip(edge_id, self.strip_width)

        # Determine side dynamically
        side = None
        for k, v in piece.edges.items():
            if v == edge_id:
                side = k
                break

        if side in ["left", "right"]:
            strip = strip.rotate(90, expand=True)

        return strip

    # Dataset interface

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]

        pieces = sample["pieces"]
        p1 = pieces[sample["idx1"]]
        p2 = pieces[sample["idx2"]]

        strip1 = self._get_canonical_strip(p1, sample["edge_id1"])
        strip2 = self._get_canonical_strip(p2, sample["edge_id2"])

        strip2 = strip2.transpose(Image.FLIP_LEFT_RIGHT)

        TARGET_WIDTH = 224

        strip1 = strip1.resize((TARGET_WIDTH, self.strip_width))
        strip2 = strip2.resize((TARGET_WIDTH, self.strip_width))

        img1 = self.transform(strip1)
        img2 = self.transform(strip2)

        label = torch.tensor(sample["label"], dtype=torch.float32)

        return img1, img2, label

