import torch
import copy
import itertools
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
from torchvision import transforms
from .model import SiameseCompatibility
from puzzle_board import PuzzleBoard, rotation_to_align, rotate_image_and_edges
from PIL import Image
from puzzle_piece import Piece_of_Puzzle, reset_id_generators
import math
from generate_puzzle_pieces import Image_Puzzle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def strip_to_tensor(strip):
    arr = np.array(strip).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def reconstruct_beam(pieces, grid_size, beam_width=5, strip_width=3):

    print(f"Starting Beam Search Reconstruction (beam={beam_width})")

    model = SiameseCompatibility().to(device)
    model.load_state_dict(torch.load("Cosine_Embedding_Loss_model.pth", map_location=device))
    model.eval()
    
    def compute_score(pieceA, sideA, pieceB, sideB):

        def get_canonical(piece, edge_id):
            strip = piece.get_edge_strip(edge_id, strip_width)

            # detect side
            side = None
            for k, v in piece.edges.items():
                if v == edge_id:
                    side = k
                    break

            # rotate vertical edges to horizontal
            if side in ["left", "right"]:
                strip = strip.rotate(90, expand=True)

            return strip

        strip1 = get_canonical(pieceA, pieceA.edges[sideA])
        strip2 = get_canonical(pieceB, pieceB.edges[sideB])

        # flip second strip exactly like training
        strip2 = strip2.transpose(Image.FLIP_LEFT_RIGHT)

        TARGET_WIDTH = 224

        strip1 = strip1.resize((TARGET_WIDTH, strip_width))
        strip2 = strip2.resize((TARGET_WIDTH, strip_width))

        t1 = transforms.ToTensor()(strip1)
        t2 = transforms.ToTensor()(strip2)

        t1 = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(t1)
        t2 = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(t2)

        t1 = t1.unsqueeze(0).to(device)
        t2 = t2.unsqueeze(0).to(device)

        with torch.no_grad():
            return model(t1, t2).item()


    # Initial empty grid
    empty_grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]

    # Anchor placement
    anchor = pieces[0]
    empty_grid[0][0] = anchor

    beam = [{
        "grid": empty_grid,
        "used": {anchor.id},
        "score": 0.0
    }]

    total_positions = grid_size * grid_size

    for step in range(1, total_positions):

        r = step // grid_size
        c = step % grid_size

        print(f"Expanding position ({r},{c})")

        candidates = []

        for state in beam:

            grid = state["grid"]
            used_ids = state["used"]
            base_score = state["score"]

            for piece in pieces:
                if piece.id in used_ids:
                    continue

                for rotation in [0, 90, 180, 270]:
                    rotated = rotate_image_and_edges(piece, rotation)

                    placement_score = 0

                    # left neighbor
                    if c > 0 and grid[r][c-1] is not None:
                        left_neighbor = grid[r][c-1]
                        placement_score += compute_score(
                            left_neighbor, "right",
                            rotated, "left"
                        )

                    # top neighbor
                    if r > 0 and grid[r-1][c] is not None:
                        top_neighbor = grid[r-1][c]
                        placement_score += compute_score(
                            top_neighbor, "bottom",
                            rotated, "top"
                        )

                    new_grid = copy.deepcopy(grid)
                    new_grid[r][c] = rotated

                    new_used = set(used_ids)
                    new_used.add(piece.id)

                    candidates.append({
                        "grid": new_grid,
                        "used": new_used,
                        "score": base_score + placement_score
                    })

        # Keep top beam_width states
        candidates.sort(key=lambda x: x["score"], reverse=True)
        beam = candidates[:beam_width]

        print(f"Beam pruned to {len(beam)} states")

    # Best final state
    best_state = beam[0]
    best_grid = best_state["grid"]

    print("Rendering final image...")

    piece_size = pieces[0].size
    final_img = Image.new("RGB",
                          (grid_size * piece_size,
                           grid_size * piece_size))

    for r in range(grid_size):
        for c in range(grid_size):
            final_img.paste(
                best_grid[r][c].image,
                (c * piece_size, r * piece_size)
            )

    final_img.show()
    final_img.save("reconstructed_beam.jpg")

    print("Beam Reconstruction Complete.")
    
def main():
    print("Starting reconstruction...")

    reset_id_generators()  

    puzzle = Image_Puzzle(
        "Forrest.jpg"
    )

    pieces = puzzle.split_into_grid(4, randomize=True)

    piece_objs = [
        Piece_of_Puzzle(img)
        for img in pieces
    ]

    reconstruct_beam(piece_objs,4,5)
    

if __name__ == "__main__":
    main()
    
    



