from google import genai
import numpy as np

API_KEY = 'AIzaSyAxsDAxxWfndet6-Mo2v2RJMF7Cd7Sht-E' 
#need a new api key this one makes u get error 429 so if u wanna test make one urself - dq 07/22/2025

# Colors
BLUE = (0,0,255)
RED = (255,0,0)
YELLOW = (255,255,0)
WHITE = (255,255,255)
BLACK = (0,0,0)

# Piece values
AI_PIECE = -1
EMPTY = 0
PLAYER_PIECE = 1

# Turn indicators (0 = player or first agent, 1 = AI or second agent)
AI = 1
PLAYER = 0

# Default board dimensions 
NUM_ROW = 6
NUM_COL = 7

# Scoring
WIN_SCORE = 100000

# Core game functions 
def createBoard():
    return np.zeros((NUM_ROW, NUM_COL), dtype=int)

def isValidMove(board, col):
    return 0 <= col < NUM_COL and board[0][col] == EMPTY

def playMove(board, row, col, piece):
    board[row][col] = piece

def getNextRow(board, col):
    for r in range(NUM_ROW - 1, -1, -1):
        if board[r][col] == EMPTY:
            return r
    return None

def checkWinningMove(board, piece):
    # Horizontal
    for r in range(NUM_ROW):
        for c in range(NUM_COL - 3):
            if all(board[r][c+i] == piece for i in range(4)):
                return True
    # Vertical
    for c in range(NUM_COL):
        for r in range(NUM_ROW - 3):
            if all(board[r+i][c] == piece for i in range(4)):
                return True
    # Positive diagonal
    for r in range(NUM_ROW - 3):
        for c in range(NUM_COL - 3):
            if all(board[r+i][c+i] == piece for i in range(4)):
                return True
    # Negative diagonal
    for r in range(3, NUM_ROW):
        for c in range(NUM_COL - 3):
            if all(board[r-i][c+i] == piece for i in range(4)):
                return True
    return False

# Heuristic Evaluation Functions 
def evaluate_window(window):
    count_p = window.count(PLAYER_PIECE)
    count_ai = window.count(AI_PIECE)
    count_emp = window.count(EMPTY)
    if count_p > 0 and count_ai > 0:
        return 0
    score = 0
    if count_ai == 0:
        if count_p == 4:
            score += WIN_SCORE
        elif count_p == 3 and count_emp == 1:
            score += 5
        elif count_p == 2 and count_emp == 2:
            score += 2
    if count_p == 0:
        if count_ai == 4:
            score -= WIN_SCORE
        elif count_ai == 3 and count_emp == 1:
            score -= 4
        elif count_ai == 2 and count_emp == 2:
            score -= 2
    return score

def evaluate_board(board):
    if checkWinningMove(board, PLAYER_PIECE):
        return WIN_SCORE
    if checkWinningMove(board, AI_PIECE):
        return -WIN_SCORE
    if not np.any(board == EMPTY):
        return 0
    score = 0
    # Center column preference
    center = list(board[:, NUM_COL//2])
    score += 3 * (center.count(PLAYER_PIECE) - center.count(AI_PIECE))
    # Windows
    for r in range(NUM_ROW):
        for c in range(NUM_COL-3):
            score += evaluate_window(list(board[r, c:c+4]))
    for c in range(NUM_COL):
        for r in range(NUM_ROW-3):
            score += evaluate_window([board[r+i][c] for i in range(4)])
    for r in range(NUM_ROW-3):
        for c in range(NUM_COL-3):
            score += evaluate_window([board[r+i][c+i] for i in range(4)])
    for r in range(3, NUM_ROW):
        for c in range(NUM_COL-3):
            score += evaluate_window([board[r-i][c+i] for i in range(4)])
    return score

# Gemini API prompt
def gemini_prompt(board, depth=None):
    """Queries the Gemini API for the best move. Ignores depth parameter."""
    client = genai.Client(api_key=API_KEY)
    prompt = (
    "You are an expert Connect Four AI. The current board state is:\n"
    f"{board}\n\n"
    "Notation: 0 = empty, 1 = opponent/human, –1 = AI.\n"
    "Evaluate all legal moves, prioritizing:\n"
    "  1. Immediate winning moves for yourself\n"
    "  2. Moves that block the opponent's imminent win\n"
    "  3. Future setup moves (forks, center control)\n\n"
    f"Which column (1–{NUM_COL}) do you choose? "
    "Reply with 10 seconds and only the 1‑based column number.\n"
    )
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    )
    return int(response.text.strip()) - 1

# Helper function to display board
def print_board(board):
    header = "".join(f" {i+1} " for i in range(NUM_COL))
    print("\n" + header)
    for row in board:
        line = "".join(
            f"[{'.' if cell==EMPTY else 'X' if cell==PLAYER_PIECE else 'O'}]"
            for cell in row
        )
        print(line)
    print()

# Utility function for random valid move
def random_valid_column(board):
    valid = [c for c in range(NUM_COL) if isValidMove(board, c)]
    import random
    return random.choice(valid) if valid else 0
