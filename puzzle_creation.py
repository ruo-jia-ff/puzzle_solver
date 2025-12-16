from PIL import Image, ImageDraw, ImageFont
import math 
import matplotlib.pyplot as plt
import random

def crop_to_square_center(image_path: str):
    """
    Crops the image to a square by truncating the longer side from the center.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: Cropped square image.
    """
    image = Image.open(image_path)
    width, height = image.size
    side_length = min(width, height)

    left = (width - side_length) // 2
    top = (height - side_length) // 2
    right = left + side_length
    bottom = top + side_length

    return image.crop((left, top, right, bottom))

def split_image_into_grid_pieces(image, n, randomize = True):
    """
    Splits the image at image_path into an n x n grid of pieces.

    Args:
        iamge (PIL): Python Image Library Image.
        n (int): Number of divisions along each axis.

    Returns:
        List of PIL.Image objects representing the pieces.
    """
    width, height = image.size
    piece_width = width // n
    piece_height = height // n

    pieces = []

    placement_dict = dict()
    for row in range(n):
        for col in range(n):
            left = col * piece_width
            upper = row * piece_height
            right = (col + 1) * piece_width
            lower = (row + 1) * piece_height
            box = (left, upper, right, lower)
            piece = image.crop(box)
            pieces.append(piece)
            placement_dict[row*n+col] = {'shuffle_id': row*n + col, 'angle': 0}

    if randomize:
        shuffled_indices = list(range(len(pieces)))
        new_pieces = []
        random.shuffle(shuffled_indices)

        for orig_id, piece in enumerate(pieces):
            shuffle_id = shuffled_indices[orig_id]
            placement_dict[shuffle_id]['shuffle_id'] = orig_id
            placement_dict[shuffle_id]['angle'] = random.choice([0, 90, 180, 270])
            new_angle = placement_dict[shuffle_id]['angle']

            new_piece = pieces[shuffle_id].rotate(new_angle, expand = True)
            new_pieces.append(new_piece)
            
        pieces = new_pieces

    return pieces, placement_dict

def show_all_pieces(pieces, 
                    dimensions = None,
                    title = None):
    """
    Displays all pieces in an n x n grid.
    """
    if dimensions:
        h, w = dimensions
    else:
        h = w = int(math.sqrt(len(pieces)))

    # Create figure 
    fig, axes = plt.subplots(h, w, figsize=(8, 8))

    if title:
        fig.suptitle(
            title, 
            fontdict={'size': 14, 'weight': 'bold', 'fontname': 'Arial'}, 
            y=1.001  # Adjust the height
        )

    for i, ax in enumerate(axes.flat):
        ax.imshow(pieces[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def reassemble_pieces(shuffled_pieces, placement_dict):

    reassembled_pieces = []
    reassembled_order = []
    for k in range(len(shuffled_pieces)):
        shuffle_id = placement_dict[k]['shuffle_id']
        inv_angle = (360 - placement_dict[k]['angle'])%360
        piece = shuffled_pieces[shuffle_id]
        original_piece = piece.rotate(inv_angle, expand=True)
        reassembled_pieces.append(original_piece)
        reassembled_order.append((shuffle_id, inv_angle))

    show_all_pieces(reassembled_pieces)

    return reassembled_order