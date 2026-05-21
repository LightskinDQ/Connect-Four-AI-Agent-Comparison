from google import genai
import connect4game
from connect4game import *
import random
import numpy as np

# Additional piece for random drops
RAND_PIECE = 9

# Gemini API prompt for random version
def gemini_prompt(board, depth=None):
    """Queries the Gemini API for the best move in random variant."""
    client = genai.Client(api_key=connect4game.API_KEY)
    prompt = (
    "You are an expert Connect Four AI. You are playing a variation where every 4 moves, there is a 25% chance of a blocker piece getting dropped into a column that will restrict you and your opponent. The current board state is:\n"
    f"{board}\n\n"
    "Notation: 0 = empty, 1 = opponent/human, –1 = AI, 9 = blocker piece.\n"
    "Evaluate all legal moves, prioritizing:\n"
    "  1. Immediate winning moves for yourself\n"
    "  2. Moves that block the opponent's imminent win\n"
    "  3. Future setup moves (forks, center control)\n"
    "  Take into account the chance of blocker pieces\n\n"
    f"Which column (1–{NUM_COL}) do you choose? "
    "Reply with 10 seconds and only the 1‑based column number.\n"
    )
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    )
    return int(response.text.strip()) - 1

# Modified evaluation functions for random pieces
def evaluate_window_with_random(window):
    """Window evaluation that treats random pieces as blockers"""
    count_p = window.count(PLAYER_PIECE)
    count_ai = window.count(AI_PIECE)
    count_emp = window.count(EMPTY)
    count_rand = window.count(RAND_PIECE)
    
    # If there are random pieces, they block potential wins
    if count_rand > 0:
        return 0
    
    # If both players have pieces in the window, it's blocked
    if count_p > 0 and count_ai > 0:
        return 0
    
    score = 0
    if count_ai == 0:  # Only player pieces
        if count_p == 4:
            score += WIN_SCORE
        elif count_p == 3 and count_emp == 1:
            score += 5
        elif count_p == 2 and count_emp == 2:
            score += 2
    
    if count_p == 0:  # Only AI pieces
        if count_ai == 4:
            score -= WIN_SCORE
        elif count_ai == 3 and count_emp == 1:
            score -= 4
        elif count_ai == 2 and count_emp == 2:
            score -= 2
    
    return score

def evaluate_board_with_random(board):
    """Modified evaluation that treats random pieces as neutral obstacles"""
    if checkWinningMove(board, PLAYER_PIECE):
        return WIN_SCORE
    if checkWinningMove(board, AI_PIECE):
        return -WIN_SCORE
    if not np.any(board == EMPTY):
        return 0
    
    score = 0
    # Center column preference (ignore random pieces)
    center = list(board[:, NUM_COL//2])
    center_filtered = [cell for cell in center if cell != RAND_PIECE]
    score += 3 * (center_filtered.count(PLAYER_PIECE) - center_filtered.count(AI_PIECE))
    
    # Windows evaluation (modified to handle random pieces)
    for r in range(NUM_ROW):
        for c in range(NUM_COL-3):
            window = list(board[r, c:c+4])
            score += evaluate_window_with_random(window)
    for c in range(NUM_COL):
        for r in range(NUM_ROW-3):
            window = [board[r+i][c] for i in range(4)]
            score += evaluate_window_with_random(window)
    for r in range(NUM_ROW-3):
        for c in range(NUM_COL-3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window_with_random(window)
    for r in range(3, NUM_ROW):
        for c in range(NUM_COL-3):
            window = [board[r-i][c+i] for i in range(4)]
            score += evaluate_window_with_random(window)
    
    return score

# Helper function to display board with random pieces
def print_board_with_random(board):
    header = "".join(f" {i+1} " for i in range(NUM_COL))
    print("\n" + header)
    for row in board:
        line = "".join(
            f"[{'.' if cell==EMPTY else 'X' if cell==PLAYER_PIECE else 'O' if cell==AI_PIECE else '#'}]"
            for cell in row
        )
        print(line)
    print()

def dropRandomPiece(board):
    """Drop a random blocker piece with 25% probability"""
    print("Chance to have a blocker piece dropped")
    if random.random() < 0.25:
        valid_cols = [col for col in range(NUM_COL) if getNextRow(board, col) is not None]
        if valid_cols:
            rand_col = random.choice(valid_cols)
            row = getNextRow(board, rand_col)
            playMove(board, row, rand_col, RAND_PIECE)
            print(f"Random piece dropped in column {rand_col + 1}!")
        else:
            print("No valid columns for random drop")

# Mode Selection 
def choose_mode(depth=4):
    """Let user pick Human vs Agent or Agent vs Agent and select agents."""
    import algorithms
    
    modes = {"1": "Human vs Agent", "2": "Agent vs Agent"}
    agents = {
        "minimax": algorithms.minimax_move,
        "alphabeta": algorithms.alpha_beta_move,
        "expectiminimax": algorithms.expectiminimax_move,
        "expectiminimax_random": algorithms.expectiminimax_random_move,  
        "gemini": gemini_prompt
    }
    print("Select mode:")
    for k,v in modes.items(): 
        print(f" {k}: {v}")
    mode = input("Enter choice: ").strip()
    
    if mode == "1":
        print("Available agents:", ", ".join(agents.keys()))
        agent_name = input("Choose agent: ").strip()
        if agent_name in agents:
            return ("human", agents[agent_name])
        else:
            print("Invalid agent, using random")
            return ("human", connect4game.random_valid_column)
    elif mode == "2":
        print("Available agents:", ", ".join(agents.keys()))
        agent1_name = input("Choose first agent: ").strip()
        agent2_name = input("Choose second agent: ").strip()
        agent1 = agents.get(agent1_name, connect4game.random_valid_column)
        agent2 = agents.get(agent2_name, connect4game.random_valid_column)
        return (agent1, agent2)
    else:
        print("Invalid mode")
        return (connect4game.random_valid_column, connect4game.random_valid_column)

def playRandomGame():
    """Play a game with random piece drops"""
    board = createBoard()
    move_count = 0
    
    agent1, agent2 = choose_mode()
    
    while True:
        print_board_with_random(board)
        
        if move_count % 2 == 0:  # Player 1 turn
            if agent1 == "human":
                col = int(input(f"Your move (1-{NUM_COL}): ")) - 1
            else:
                col = agent1(board, 4)
            piece = PLAYER_PIECE
        else:  # Player 2 turn
            if agent2 == "human":
                col = int(input(f"Your move (1-{NUM_COL}): ")) - 1
            else:
                col = agent2(board, 4)
            piece = AI_PIECE
        
        if isValidMove(board, col):
            row = getNextRow(board, col)
            playMove(board, row, col, piece)
            
            if checkWinningMove(board, piece):
                print_board_with_random(board)
                winner = "Player 1" if piece == PLAYER_PIECE else "Player 2"
                print(f"{winner} wins!")
                break
            
            if not np.any(board == EMPTY):
                print_board_with_random(board)
                print("It's a draw!")
                break
            
            move_count += 1
            
            # Check for random drop every 4 moves
            if move_count % 4 == 0:
                dropRandomPiece(board)
        else:
            print("Invalid move, try again")

if __name__ == '__main__':
    playRandomGame()
