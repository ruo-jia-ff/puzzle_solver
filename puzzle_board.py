import matplotlib.pyplot as plt
from collections import defaultdict, deque
from copy import deepcopy
from puzzle_piece import Piece_of_Puzzle
from PIL import Image

def rotation_to_align(current_side, reference_side):
    """Calculate rotation needed to align current_side opposite to reference_side"""
    sides = ["top", "right", "bottom", "left"]
    idx_curr = sides.index(current_side)
    idx_ref = sides.index(reference_side)
    # We want current_side to face opposite of reference_side
    delta = (idx_ref + 2 - idx_curr) % 4
    return delta * 90

def rotate_edges(edges, rotation_degrees):
    """Rotate edge labels (IDs) by 90, 180, or 270 degrees clockwise."""
    order = ["top", "right", "bottom", "left"]
    steps = (rotation_degrees // 90) % 4
    rotated = {}
    for i, side in enumerate(order):
        rotated[order[(i + steps) % 4]] = edges[side]
    return rotated

def rotate_image_and_edges(piece, degrees):
    """Rotate a puzzle piece's image and edges."""
    rotated_img = piece.image.rotate(-degrees, expand=True)  # PIL rotates counterclockwise by default
    rotated_edges = rotate_edges(piece.edges, degrees)
    return Piece_of_Puzzle(rotated_img, id = piece.id, edges=rotated_edges)

class PuzzleBoard:
    def __init__(self, anchor_edge=None, anchor_direction='right'):
        self.pieces = dict()          # {piece_id: Piece_of_Puzzle}
        self.edge_map = dict()        # {edge_id: (piece_id, side)}
        self.edge_pairs = set()       # {(edge_id1, edge_id2)}
        self.placed_pieces = {}       # {piece_id: (position, rotated_piece)}
        self.occupied_positions = {}  # {(x, y): piece_id}
        self.chains = []
        self.piece_to_chain = {}
        self.anchor_edge = anchor_edge
        self.anchor_direction = anchor_direction

    def add_piece(self, piece):
        """Adds a single Piece_of_Puzzle and registers its edges."""
        self.pieces[piece.id] = piece
        for side, edge_id in piece.edges.items():
            if edge_id is not None:
                self.edge_map[edge_id] = (piece.id, side)
        return piece.id
    
    def list_all_edges(self):
        """Lists all edges currently registered in the puzzle."""
        edge_list = []
        for edge_id, (piece_id, side) in sorted(self.edge_map.items()):
            edge_list.append((edge_id, piece_id, side))
        edge_list.sort(key=lambda x: (x[1], x[0]))
        return edge_list
    
    def _check_collision(self, edge_id1, edge_id2):
        """
        Check if adding this edge pair would cause a collision.
        Returns (is_valid, error_message)
        """
        if edge_id1 not in self.edge_map or edge_id2 not in self.edge_map:
            return False, f"Edge(s) not found: {edge_id1}, {edge_id2}"
        
        piece_id1, side1 = self.edge_map[edge_id1]
        piece_id2, side2 = self.edge_map[edge_id2]
        
        if piece_id1 == piece_id2:
            return False, "Cannot connect a piece to itself"
        
        # If neither piece is placed yet, it's valid
        if piece_id1 not in self.placed_pieces and piece_id2 not in self.placed_pieces:
            return True, None
        
        # If both pieces are already placed, check if they're adjacent
        if piece_id1 in self.placed_pieces and piece_id2 in self.placed_pieces:
            pos1, rotated_piece1 = self.placed_pieces[piece_id1]
            pos2, _ = self.placed_pieces[piece_id2]
                        
            # Check if they're adjacent
            direction_delta = {
                "top": (0, -1), "bottom": (0, 1), 
                "left": (-1, 0), "right": (1, 0)
            }
            
            # Find which side of rotated_piece1 has edge_id1
            actual_side1 = None
            for s, e in rotated_piece1.edges.items():
                if e == edge_id1:
                    actual_side1 = s
                    break
            
            # Calculate expected position of piece2 based on piece1's side
            if actual_side1:
                dx, dy = direction_delta[actual_side1]
                expected_pos2 = (pos1[0] + dx, pos1[1] + dy)
                
                if pos2 != expected_pos2:
                    return False, f"Pieces already placed but not adjacent: {piece_id1} at {pos1}, {piece_id2} at {pos2}"
            
            return True, None
        
        # One piece is placed, one is not - simulate placement
        if piece_id1 in self.placed_pieces:
            ref_id = piece_id1
            ref_pos, ref_piece = self.placed_pieces[piece_id1]
        else:
            ref_id= piece_id2
            ref_pos, ref_piece = self.placed_pieces[piece_id2]
        
        # Find which side of the rotated reference piece has the reference edge
        actual_ref_side = None
        for s, e in ref_piece.edges.items():
            if e == edge_id1 if ref_id == piece_id1 else e == edge_id2:
                actual_ref_side = s
                break
        
        if not actual_ref_side:
            return False, f"Edge not found on rotated piece {ref_id}"
        
        # Calculate new position
        direction_delta = {
            "top": (0, -1), "bottom": (0, 1), 
            "left": (-1, 0), "right": (1, 0)
        }
        dx, dy = direction_delta[actual_ref_side]
        new_pos = (ref_pos[0] + dx, ref_pos[1] + dy)
        
        # Check if position is occupied
        if new_pos in self.occupied_positions:
            return False, f"Position {new_pos} already occupied by piece {self.occupied_positions[new_pos]}"
        
        return True, None
    
    def add_edge_pair_to_chain(self, edge_id1, edge_id2, overwrite=False):
        """
        Adds an edge connection with collision detection.
        Will automatically rotate pieces as needed.
        """
        # Check for collision
        is_valid, error_msg = self._check_collision(edge_id1, edge_id2)
        
        if not is_valid and not overwrite:
            print(f"Cannot add connection: {error_msg}")
            return False
        
        piece_id1, side1 = self.edge_map[edge_id1]
        piece_id2, side2 = self.edge_map[edge_id2]
        
        # Check if either edge is already part of a pair
        existing = {pair for pair in self.edge_pairs if edge_id1 in pair or edge_id2 in pair}
        
        if existing and not overwrite:
            print(f"Edge already in use: {existing}. Use overwrite=True to replace.")
            return False
        elif existing and overwrite:
            self.edge_pairs.difference_update(existing)
            self._rebuild_placement()
        
        # Add edge pair
        self.edge_pairs.add(tuple(sorted((edge_id1, edge_id2))))
        
        # Update placement
        self._rebuild_placement()
        
        # Update chains
        c1 = self._get_chain_index(piece_id1)
        c2 = self._get_chain_index(piece_id2)
        
        if c1 is not None and c2 is not None:
            if c1 != c2:
                self.chains[c1].update(self.chains[c2])
                for pid in self.chains[c2]:
                    self.piece_to_chain[pid] = c1
                del self.chains[c2]
        elif c1 is not None:
            self.chains[c1].add(piece_id2)
            self.piece_to_chain[piece_id2] = c1
        elif c2 is not None:
            self.chains[c2].add(piece_id1)
            self.piece_to_chain[piece_id1] = c2
        else:
            new_index = len(self.chains)
            self.chains.append(set([piece_id1, piece_id2]))
            self.piece_to_chain[piece_id1] = new_index
            self.piece_to_chain[piece_id2] = new_index
        
        print(f"Success! Connected {edge_id1} (piece {piece_id1}, {side1}) to {edge_id2} (piece {piece_id2}, {side2})")
        return True
    
    def _rebuild_placement(self):
        """Rebuild placement map from scratch based on current edge_pairs"""
        self.placed_pieces = {}
        self.occupied_positions = {}
        
        if not self.edge_pairs:
            return
        
        # Build adjacency list
        adj = defaultdict(list)
        for e1, e2 in self.edge_pairs:
            if e1 in self.edge_map and e2 in self.edge_map:
                id1, side1 = self.edge_map[e1]
                id2, side2 = self.edge_map[e2]
                adj[id1].append((id2, side1, side2, e1, e2))
                adj[id2].append((id1, side2, side1, e2, e1))
        
        # Find all connected components and place them
        visited = set()
        
        for start_id in adj.keys():
            if start_id in visited:
                continue
            
            # Place first piece at origin
            first_piece = self.pieces[start_id]
            self.placed_pieces[start_id] = ((0, 0), first_piece)
            self.occupied_positions[(0, 0)] = start_id
            
            # BFS to place connected pieces
            queue = deque([start_id])
            visited.add(start_id)
            
            while queue:
                current_id = queue.popleft()
                current_pos, current_piece = self.placed_pieces[current_id]
                
                for neighbor_id, _, side_to, edge_from, _ in adj[current_id]:
                    if neighbor_id in visited:
                        continue
                    
                    visited.add(neighbor_id)
                    
                    # Find actual side on rotated current piece
                    actual_side_from = None
                    for s, e in current_piece.edges.items():
                        if e == edge_from:
                            actual_side_from = s
                            break
                    
                    if not actual_side_from:
                        continue
                    
                    # Calculate rotation for neighbor
                    neighbor_piece = self.pieces[neighbor_id]
                    degrees = rotation_to_align(side_to, actual_side_from)
                    rotated_neighbor = rotate_image_and_edges(neighbor_piece, degrees)
                    
                    # Calculate position
                    direction_delta = {
                        "top": (0, -1), "bottom": (0, 1),
                        "left": (-1, 0), "right": (1, 0)
                    }
                    dx, dy = direction_delta[actual_side_from]
                    new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                    
                    # Place piece
                    self.placed_pieces[neighbor_id] = (new_pos, rotated_neighbor)
                    self.occupied_positions[new_pos] = neighbor_id
                    queue.append(neighbor_id)
    
    def remove_edge_from_chain(self, edge_id):
        """Removes all edge pairs involving the given edge ID"""
        to_remove = {pair for pair in self.edge_pairs if edge_id in pair}
        
        if not to_remove:
            print(f"No edge pairs found involving edge {edge_id}.")
            return
        
        self.edge_pairs.difference_update(to_remove)
        self._rebuild_placement()
        self._rebuild_chains()
    
    def _rebuild_chains(self):
        """Recomputes chains from scratch based on current edge_pairs."""
        self.chains = []
        self.piece_to_chain = {}
        graph = defaultdict(set)
        
        for edge_id1, edge_id2 in self.edge_pairs:
            if edge_id1 in self.edge_map and edge_id2 in self.edge_map:
                id1, _ = self.edge_map[edge_id1]
                id2, _ = self.edge_map[edge_id2]
                graph[id1].add(id2)
                graph[id2].add(id1)
        
        visited = set()
        
        def dfs(node, component):
            visited.add(node)
            component.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in graph:
            if node not in visited:
                comp = set()
                dfs(node, comp)
                chain_idx = len(self.chains)
                self.chains.append(comp)
                for pid in comp:
                    self.piece_to_chain[pid] = chain_idx
    
    def _get_chain_index(self, piece_id):
        return self.piece_to_chain.get(piece_id)
    
    def list_edge_connections(self):
        """Returns a list of current edge-to-edge connections."""
        connections = []
        for edge1, edge2 in sorted(self.edge_pairs):
            if edge1 in self.edge_map and edge2 in self.edge_map:
                id1, side1 = self.edge_map[edge1]
                id2, side2 = self.edge_map[edge2]
                connections.append(((edge1, id1, side1), (edge2, id2, side2)))
        return connections
    
    def visualize_chains(self, fontsize=10):
        """Visualize all connected components"""
        if not self.placed_pieces:
            print("No pieces placed yet.")
            return
        
        # Group by connected component
        components = defaultdict(list)
        for piece_id, (pos, piece) in self.placed_pieces.items():
            chain_idx = self.piece_to_chain.get(piece_id, -1)
            components[chain_idx].append((piece_id, pos, piece))
        
        for chain_idx, pieces_data in components.items():
            print(f"\n Visualizing component #{chain_idx + 1} with {len(pieces_data)} pieces")
            
            # Normalize positions
            all_positions = [pos for _, pos, _ in pieces_data]
            min_x = min(x for x, y in all_positions)
            min_y = min(y for x, y in all_positions)
            
            # Create final image
            piece_size = self.pieces[pieces_data[0][0]].size
            max_x = max(x for x, y in all_positions) - min_x
            max_y = max(y for x, y in all_positions) - min_y
            
            num_cols = max_x + 1
            num_rows = max_y + 1
            final_img = Image.new("RGB", (num_cols * piece_size, num_rows * piece_size))
            
            _, ax = plt.subplots(figsize=(num_cols * 2, num_rows * 2))
            ax.axis('off')
            
            for piece_id, (orig_x, orig_y), piece in pieces_data:
                x = orig_x - min_x
                y = orig_y - min_y
                px = x * piece_size
                py = y * piece_size
                
                final_img.paste(piece.image, (px, py))
                
                w, h = piece.image.size
                if piece.edges["top"]:
                    ax.text(px + w / 2, py + 5, f"T:{piece.edges['top']}", 
                           ha='center', va='top', fontsize=fontsize, color='white')
                if piece.edges["right"]:
                    ax.text(px + w - 5, py + h / 2, f"R:{piece.edges['right']}", 
                           ha='right', va='center', fontsize=fontsize, color='white')
                if piece.edges["bottom"]:
                    ax.text(px + w / 2, py + h - 5, f"B:{piece.edges['bottom']}", 
                           ha='center', va='bottom', fontsize=fontsize, color='white')
                if piece.edges["left"]:
                    ax.text(px + 5, py + h / 2, f"L:{piece.edges['left']}", 
                           ha='left', va='center', fontsize=fontsize, color='white')
                
                ax.text(px + w / 2, py + h / 2 - 5, f"ID:{piece.id}", 
                       ha='center', va='top', fontsize=fontsize, color='black')
            
            ax.imshow(final_img)
            plt.tight_layout()
            plt.show()
