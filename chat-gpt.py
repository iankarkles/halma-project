import random
import time

class HalmaGame:
    def __init__(self):
        # Define the game state
        self.board = [[0]*8 for _ in range(8)]
        self.board[0][:3] = [1]*3
        self.board[-1][-3:] = [2]*3
        self.player = 1

    def get_legal_moves(self, player):
        # Define the legal moves
        moves = []
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == player:
                    for dr, dc in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                        r2, c2 = r+dr, c+dc
                        if 0<=r2<8 and 0<=c2<8 and self.board[r2][c2] == 0:
                            moves.append((r,c,r2,c2))
                        elif 0<=r2<8 and 0<=c2<8 and self.board[r2][c2] != player:
                            r3, c3 = r2+dr, c2+dc
                            if 0<=r3<8 and 0<=c3<8 and self.board[r3][c3] == 0:
                                moves.append((r,c,r3,c3))
        return moves

    def apply_move(self, move):
        # Apply the move to the game state
        r1, c1, r2, c2 = move
        self.board[r1][c1] = 0
        self.board[r2][c2] = self.player
        if r2 == 0 and self.player == 1:
            # Player 1 has won
            return 1
        elif r2 == 7 and self.player == 2:
            # Player 2 has won
            return 2
        else:
            # Switch players
            self.player = 3 - self.player
            return 0

    def get_winner(self):
        # Check if there's a winner
        if all(self.board[0][:3]):
            return 2
        elif all(self.board[-1][-3:]):
            return 1
        else:
            return 0

class HalmaAI:
    def __init__(self, player):
        self.player = player

    def minimax(self, game, depth, alpha, beta, maximizing_player):
        # Implement the minimax algorithm with alpha-beta pruning
        if depth == 0 or game.get_winner():
            return game.get_winner()

        legal_moves = game.get_legal_moves(self.player)
        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                game_copy = copy.deepcopy(game)
                game_copy.apply_move(move)
                eval = self.minimax(game_copy, depth-1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                game_copy = copy.deepcopy(game)
                game_copy.apply_move(move)
                eval = self.minimax(game_copy, depth-1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_next_move(self, game):
        # Implement the search algorithm that calls the minimax algorithm
        legal_moves = game.get_legal_moves(self.player)
        best_move = None
        if legal_moves:
            if len(legal_moves) == 1:
                return legal_moves[0]
            best_eval = float('-inf')
            for move in legal_moves:
                game_copy = copy.deepcopy(game)
                game_copy.apply_move(move)
                eval = self.minimax(game_copy, 4, float('-inf'), float('inf'), False)
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
        return best_move

"""
This code defines a HalmaGame class that represents the game state, and a HalmaAI class that implements the minimax algorithm with alpha-beta pruning to find the optimal move for the AI player. The HalmaAI class has a get_next_move method that calls the minimax algorithm with a depth of 4 to search for the optimal move. The depth of the search can be adjusted to trade off between performance and accuracy. The HalmaGame class has methods to get the legal moves for a player, apply a move to the game state, and check if there's a winner.
"""
class HalmaGame:
    def __init__(self):
        self.board = [[0] * 8 for _ in range(8)]
        for i in range(3):
            for j in range(8):
                if i % 2 == j % 2:
                    self.board[i][j] = 1
        for i in range(5, 8):
            for j in range(8):
                if i % 2 == j % 2:
                    self.board[i][j] = 2
        self.player = 1

    def get_legal_moves(self, player):
        moves = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == player:
                    for r in range(i-2, i+3):
                        for c in range(j-2, j+3):
                            if r >= 0 and r < 8 and c >= 0 and c < 8 and self.board[r][c] == 0:
                                if abs(r-i) == 2 or abs(c-j) == 2:
                                    # Jump move
                                    if self.board[(i+r)//2][(j+c)//2] == 3-player:
                                        moves.append((i, j, r, c))
                                else:
                                    # Regular move
                                    moves.append((i, j, r, c))
        return moves

"""
The implementation of the get_legal_moves method in the HalmaGame class is incorrect. The current implementation checks all positions within a 5x5 grid centered around the current position of the player, which is not correct. The correct implementation should check all positions within a 2-square radius of the current position of the player. To fix the implementation, the loops for r and c should be adjusted to loop over the range (i-2, i+3) and (j-2, j+3), respectively, and an additional check should be added to ensure that the distance between the current position and the position being checked is at most 2.
Here's the corrected implementation of the get_legal_moves method:

def get_legal_moves(self, player):
    moves = []
    for i in range(8):
        for j in range(8):
            if self.board[i][j] == player:
                for r in range(i-2, i+3):
                    for c in range(j-2, j+3):
                        if abs(r-i) + abs(c-j) <= 2 and 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == 0:
                            moves.append((i, j, r, c))
    return moves


other option:
 def get_legal_moves(self, player):
        moves = []
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == player:
                    for r in range(i-2, i+3):
                        for c in range(j-2, j+3):
                            if 0 <= r <= 7 and 0 <= c <= 7 and self.board[r][c] == 0:
                                moves.append((i, j, r, c))
        return moves



"""

    def apply_move(self, move):
        r1, c1, r2, c2 = move
        self.board[r1][c1] = 0
        self.board[r2][c2] = self.player
        if r2 == 0 and self.player == 1:
            # Player 1 has won
            return 1
        elif r2 == 7 and self.player == 2:
            # Player 2 has won
            return 2
        else:
            # Switch players
            self.player = 3 - self.player
            return 0

    def get_winner(self):
        # Check if there's a winner
        if all(self.board[0][:3]):
            return 2
        elif all(self.board[-1][-3:]):
            return 1
        else:
            return 0

class HalmaAI:
    def __init__(self, player, skill_level=1):
        self.player = player
        self.skill_level = skill_level

    def evaluate_board(self, game):
        # Evaluation function that simply counts the number of pieces on the board
        p1_pieces = 0
        p2_pieces = 0
        for i in range(8):
            for j in range(8):
                if game.board[i][j] == 1:
                    p1_pieces += 1
                elif game.board[i][j] == 2:
                    p2_pieces += 1
        return p2_pieces - p1_pieces

    def minimax(self, game, depth, alpha, beta, maximizing_player):
        # Implement the minimax algorithm with alpha-beta pruning
        if depth == 0 or game.get_winner():
            return self.evaluate_board(game)

        legal_moves = game.get_legal_moves(self.player)
        if maximizing_player:
            max_eval = float('-inf')
            if self.skill_level == 1:
                # Randomize order of moves
                random.shuffle(legal_moves)
            elif self.skill_level == 2:
                # Sort moves by their evaluation score
                legal_moves.sort(key=lambda move: self.evaluate_board(game.apply_move(move)))
            for move in legal_moves:
                game_copy = HalmaGame()
                game_copy.board = [row[:] for row in game.board]
                game_copy.player = game.player
                game_copy.apply_move(move)
                eval = self.minimax(game_copy, depth-1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            if self.skill_level == 1:
                # Randomize order of moves
                random.shuffle(legal_moves)
            elif self.skill_level == 2:
                # Sort moves by their evaluation score
                legal_moves.sort(key=lambda move: self.evaluate_board(game.apply_move(move)))
            for move in legal_moves:
                game_copy = HalmaGame()
                game_copy.board = [row[:] for row in game.board]
                game_copy.player = game.player
                game_copy.apply_move(move)
                eval = self.minimax(game_copy, depth-1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move_minimax(self, game):
    # Use the minimax algorithm to find the best move for the current player
    legal_moves = game.get_legal_moves(self.player)
    best_move = None
    if self.skill_level == 0:
        # Choose random move
        best_move = random.choice(legal_moves)
    else:
        max_eval = float('-inf')
        for move in legal_moves:
            game_copy = HalmaGame()
            game_copy.board = [row[:] for row in game.board]
            game_copy.player = game.player
            game_copy.apply_move(move)
            eval = self.minimax(game_copy, depth=self.skill_level)
            if eval > max_eval:
                max_eval = eval
                best_move = move
    return best_move


def minimax(self, game, depth):
    """Minimax algorithm with alpha-beta pruning"""
    if depth == 0 or game.is_winner(self.player) or game.is_loser(self.player):
        return self.score(game)
    
    legal_moves = game.get_legal_moves()
    if game.active_player == self.player:  # maximize
        max_eval = float('-inf')
        for move in legal_moves:
            game_copy = HalmaGame()
            game_copy.board = [row[:] for row in game.board]
            game_copy.player = game.player
            game_copy.apply_move(move)
            eval = self.minimax(game_copy, depth-1)
            max_eval = max(max_eval, eval)
        return max_eval
    else:  # minimize
        min_eval = float('inf')
        for move in legal_moves:
            game_copy = HalmaGame()
            game_copy.board = [row[:] for row in game.board]
            game_copy.player = game.player
            game_copy.apply_move(move)
            eval = self.minimax(game_copy, depth-1)
            min_eval = min(min_eval, eval)
        return min_eval

def score(self, game):
    """Compute the heuristic value of a game state"""
    own_pieces = 0
    opp_pieces = 0
    for row in range(game.height):
        for col in range(game.width):
            piece = game.board[row][col]
            if piece == self.player:
                own_pieces += 1
            elif piece != '.':
                opp_pieces += 1

    if own_pieces == 0:
        return float('-inf')

    if opp_pieces == 0:
        return float('inf')

    own_center_distance = sum(min(abs(row - game.center) + abs(col - game.center), game.max_distance) 
                               for row in range(game.height) for col in range(game.width) if game.board[row][col] == self.player)

    opp_center_distance = sum(min(abs(row - game.center) + abs(col - game.center), game.max_distance) 
                               for row in range(game.height) for col in range(game.width) if game.board[row][col] != '.' and game.board[row][col] != self.player)

    return own_pieces - opp_pieces + own_center_distance - opp_center_distance / 2.0  # try to stay closer to the center and capture more opponent pieces
