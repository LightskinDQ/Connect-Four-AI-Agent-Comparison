
import connect4game
import randomConnectFour
import random
import numpy as np

# Shorthand constants
EMPTY = connect4game.EMPTY
PLAYER = connect4game.PLAYER_PIECE
AI = connect4game.AI_PIECE

# Global variable for node counting
node_count = 0

# Minimax Algorithm 
def minimax_value(board, depth, maximizing):
    global node_count
    node_count += 1
    
    if depth == 0 or connect4game.checkWinningMove(board, PLAYER) or connect4game.checkWinningMove(board, AI) or not np.any(board == EMPTY):
        return connect4game.evaluate_board(board)
    
    if maximizing:
        max_eval = -float('inf')
        for col in range(connect4game.NUM_COL):
            if connect4game.isValidMove(board, col):
                row = connect4game.getNextRow(board, col)
                temp = board.copy()
                connect4game.playMove(temp, row, col, PLAYER)
                eval_val = minimax_value(temp, depth-1, False)
                max_eval = max(max_eval, eval_val)
        return max_eval
    else:
        min_eval = float('inf')
        for col in range(connect4game.NUM_COL):
            if connect4game.isValidMove(board, col):
                row = connect4game.getNextRow(board, col)
                temp = board.copy()
                connect4game.playMove(temp, row, col, AI)
                eval_val = minimax_value(temp, depth-1, True)
                min_eval = min(min_eval, eval_val)
        return min_eval

def minimax_move(board, depth=4):
    global node_count
    node_count = 0
    
    maximizing = (np.count_nonzero(board) % 2 == 0)
    best_val = -float('inf') if maximizing else float('inf')
    best_col = None
    piece = PLAYER if maximizing else AI
    
    for col in range(connect4game.NUM_COL):
        if connect4game.isValidMove(board, col):
            row = connect4game.getNextRow(board, col)
            temp = board.copy()
            connect4game.playMove(temp, row, col, piece)
            val = minimax_value(temp, depth-1, not maximizing)
            if (maximizing and val > best_val) or (not maximizing and val < best_val):
                best_val, best_col = val, col
    
    if best_col is None:
        return random.choice([c for c in range(connect4game.NUM_COL) if connect4game.isValidMove(board, c)])
    return best_col

# Alpha-Beta Algorithm 
def alpha_beta_value(board, depth, alpha, beta, maximizing):
    global node_count
    node_count += 1
    
    if depth == 0 or connect4game.checkWinningMove(board, PLAYER) or connect4game.checkWinningMove(board, AI) or not np.any(board == EMPTY):
        return connect4game.evaluate_board(board)
    
    if maximizing:
        value = -float('inf')
        for col in range(connect4game.NUM_COL):
            if connect4game.isValidMove(board, col):
                row = connect4game.getNextRow(board, col)
                temp = board.copy()
                connect4game.playMove(temp, row, col, PLAYER)
                value = max(value, alpha_beta_value(temp, depth-1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        return value
    else:
        value = float('inf')
        for col in range(connect4game.NUM_COL):
            if connect4game.isValidMove(board, col):
                row = connect4game.getNextRow(board, col)
                temp = board.copy()
                connect4game.playMove(temp, row, col, AI)
                value = min(value, alpha_beta_value(temp, depth-1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break
        return value

def alpha_beta_move(board, depth=4):
    global node_count
    node_count = 0
    
    maximizing = (np.count_nonzero(board) % 2 == 0)
    best_val = -float('inf') if maximizing else float('inf')
    best_col = None
    alpha, beta = -float('inf'), float('inf')
    
    for col in range(connect4game.NUM_COL):
        if connect4game.isValidMove(board, col):
            row = connect4game.getNextRow(board, col)
            temp = board.copy()
            piece = PLAYER if maximizing else AI
            connect4game.playMove(temp, row, col, piece)
            val = alpha_beta_value(temp, depth-1, alpha, beta, not maximizing)
            if (maximizing and val > best_val) or (not maximizing and val < best_val):
                best_val, best_col = val, col
            alpha = max(alpha, best_val) if maximizing else alpha
            beta = min(beta, best_val) if not maximizing else beta
            if beta <= alpha:
                break
    
    if best_col is None:
        return random.choice([c for c in range(connect4game.NUM_COL) if connect4game.isValidMove(board, c)])
    return best_col

# Standard Expectiminimax Algorithm 
def expectiminimax_value(board, depth, node_type, move_probabilities=None):
    """Standard expectiminimax algorithm without random drops"""
    global node_count
    node_count += 1
    
    if depth == 0 or connect4game.checkWinningMove(board, PLAYER) or connect4game.checkWinningMove(board, AI) or not np.any(board == EMPTY):
        return connect4game.evaluate_board(board)
    
    valid_moves = [col for col in range(connect4game.NUM_COL) if connect4game.isValidMove(board, col)]
    if not valid_moves:
        return connect4game.evaluate_board(board)
    
    if node_type == 'max':
        max_v = -float('inf')
        for col in valid_moves:
            row = connect4game.getNextRow(board, col)
            temp = board.copy()
            connect4game.playMove(temp, row, col, PLAYER)
            val = expectiminimax_value(temp, depth-1, 'chance', move_probabilities)
            max_v = max(max_v, val)
        return max_v
    
    elif node_type == 'min':
        min_v = float('inf')
        for col in valid_moves:
            row = connect4game.getNextRow(board, col)
            temp = board.copy()
            connect4game.playMove(temp, row, col, AI)
            val = expectiminimax_value(temp, depth-1, 'chance', move_probabilities)
            min_v = min(min_v, val)
        return min_v
    
    else:  # chance node
        if move_probabilities is None:
            prob_per_move = 1.0 / len(valid_moves)
            move_probabilities = {col: prob_per_move for col in valid_moves}
        
        expected_value = 0.0
        current_player = PLAYER if (np.count_nonzero(board) % 2) == 0 else AI
        next_node_type = 'min' if current_player == PLAYER else 'max'
        
        for col in valid_moves:
            prob = move_probabilities.get(col, 1.0 / len(valid_moves))
            row = connect4game.getNextRow(board, col)
            temp = board.copy()
            connect4game.playMove(temp, row, col, current_player)
            value = expectiminimax_value(temp, depth-1, next_node_type, move_probabilities)
            expected_value += prob * value
        
        return expected_value

def expectiminimax_move(board, depth=4, move_probabilities=None):
    """Standard expectiminimax move selection"""
    global node_count
    node_count = 0
    
    root = 'max' if (np.count_nonzero(board) % 2) == 0 else 'min'
    best_val = -float('inf') if root == 'max' else float('inf')
    best_col = None
    piece = PLAYER if root == 'max' else AI
    
    for col in range(connect4game.NUM_COL):
        if connect4game.isValidMove(board, col):
            row = connect4game.getNextRow(board, col)
            temp = board.copy()
            connect4game.playMove(temp, row, col, piece)
            val = expectiminimax_value(temp, depth-1, 'chance', move_probabilities)
            
            if (root == 'max' and val > best_val) or (root == 'min' and val < best_val):
                best_val, best_col = val, col
    
    return best_col if best_col is not None else random.choice([c for c in range(connect4game.NUM_COL) if connect4game.isValidMove(board, c)])

# Expectiminimax with Random Drops
def expectiminimax_random_value(board, depth, node_type, move_probabilities=None, random_drop_prob=0.25):
    """Expectiminimax that accounts for random piece drops"""
    global node_count
    node_count += 1
    
    if depth == 0 or connect4game.checkWinningMove(board, PLAYER) or connect4game.checkWinningMove(board, AI) or not np.any(board == EMPTY):
        return randomConnectFour.evaluate_board_with_random(board)
    
    valid_moves = [col for col in range(connect4game.NUM_COL) if connect4game.isValidMove(board, col)]
    if not valid_moves:
        return randomConnectFour.evaluate_board_with_random(board)
    
    if node_type == 'max':
        max_v = -float('inf')
        for col in valid_moves:
            row = connect4game.getNextRow(board, col)
            temp = board.copy()
            connect4game.playMove(temp, row, col, PLAYER)
            val = expectiminimax_random_with_drop(temp, depth-1, 'min', move_probabilities, random_drop_prob)
            max_v = max(max_v, val)
        return max_v
    
    elif node_type == 'min':
        min_v = float('inf')
        for col in valid_moves:
            row = connect4game.getNextRow(board, col)
            temp = board.copy()
            connect4game.playMove(temp, row, col, AI)
            val = expectiminimax_random_with_drop(temp, depth-1, 'max', move_probabilities, random_drop_prob)
            min_v = min(min_v, val)
        return min_v
    
    else:  # chance node
        if move_probabilities is None:
            prob_per_move = 1.0 / len(valid_moves)
            move_probabilities = {col: prob_per_move for col in valid_moves}
        
        expected_value = 0.0
        current_player = PLAYER if (np.count_nonzero(board) % 2) == 0 else AI
        next_node_type = 'min' if current_player == PLAYER else 'max'
        
        for col in valid_moves:
            prob = move_probabilities.get(col, 1.0 / len(valid_moves))
            row = connect4game.getNextRow(board, col)
            temp = board.copy()
            connect4game.playMove(temp, row, col, current_player)
            value = expectiminimax_random_value(temp, depth-1, next_node_type, move_probabilities, random_drop_prob)
            expected_value += prob * value
        
        return expected_value

def expectiminimax_random_with_drop(board, depth, next_node_type, move_probabilities=None, random_drop_prob=0.25):
    """Handle the random drop chance after each move"""
    expected_value = 0.0
    
    # Case 1: No random drop
    no_drop_prob = 1.0 - random_drop_prob
    no_drop_value = expectiminimax_random_value(board, depth, next_node_type, move_probabilities, random_drop_prob)
    expected_value += no_drop_prob * no_drop_value
    
    # Case 2: Random drop occurs
    if random_drop_prob > 0:
        drop_valid_cols = [col for col in range(connect4game.NUM_COL) if connect4game.getNextRow(board, col) is not None]
        
        if drop_valid_cols:
            drop_prob_per_col = random_drop_prob / len(drop_valid_cols)
            
            for drop_col in drop_valid_cols:
                drop_row = connect4game.getNextRow(board, drop_col)
                temp_with_drop = board.copy()
                connect4game.playMove(temp_with_drop, drop_row, drop_col, randomConnectFour.RAND_PIECE)
                
                drop_value = expectiminimax_random_value(temp_with_drop, depth, next_node_type, move_probabilities, random_drop_prob)
                expected_value += drop_prob_per_col * drop_value
        else:
            expected_value += random_drop_prob * no_drop_value
    
    return expected_value

def expectiminimax_random_move(board, depth=4, move_probabilities=None, random_drop_prob=0.25):
    """Expectiminimax move selection accounting for random drops"""
    global node_count
    node_count = 0
    
    root = 'max' if (np.count_nonzero(board) % 2) == 0 else 'min'
    best_val = -float('inf') if root == 'max' else float('inf')
    best_col = None
    piece = PLAYER if root == 'max' else AI
    
    for col in range(connect4game.NUM_COL):
        if connect4game.isValidMove(board, col):
            row = connect4game.getNextRow(board, col)
            temp = board.copy()
            connect4game.playMove(temp, row, col, piece)
            
            val = expectiminimax_random_with_drop(temp, depth-1, 'min' if root=='max' else 'max', move_probabilities, random_drop_prob)
            
            if (root == 'max' and val > best_val) or (root == 'min' and val < best_val):
                best_val, best_col = val, col
    
    return best_col if best_col is not None else random.choice([c for c in range(connect4game.NUM_COL) if connect4game.isValidMove(board, c)])

# Gemini Agent wrapper
def gemini_agent(board, depth=None):
    """Wrapper for Gemini API"""
    return connect4game.gemini_prompt(board, depth)

# [Rest of testing functions remain the same but updated to use correct imports]
# Testing and Analysis functions would continue here with proper imports...
