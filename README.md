## Connect Four AI Agent Comparison

## Overview

This project implements **Connect Four** as an adversarial game with an optional stochastic twist. It supports Human vs AI and AI vs AI matchups, with multiple AI agents — including Minimax, Alpha-Beta Pruning, ExpectiMinimax, a Random agent, and Google's Gemini LLM.

The game features a **Flask** backend (Python) for game logic and AI decision-making, and a **ReactJS** frontend for interactive play.

---

## Features

- Standard Connect Four with legal move enforcement and win/draw detection
- **Stochastic mode:** a `dropRandomPiece()` function drops blocking pieces with 25% probability after every few turns
- **Multiple AI agents:** Minimax, Alpha-Beta Pruning, ExpectiMinimax, Random, and Gemini (LLM-based)
- Human vs AI and AI vs AI modes
- Decoupled architecture — swap AI agents without changing the UI

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | ReactJS |
| Backend | Python, Flask |
| AI (external) | Google Gemini 2.5 Pro API |
| Data handling | NumPy |

---

## Project Structure

```
project/
├── app.py                   # Flask entry point & API endpoints
├── connect4game.py          # Core game logic (board, moves, win detection)
├── algorithms.py            # Minimax, Alpha-Beta, ExpectiMinimax
├── randomConnectFour.py     # Random agent
│── test1.py                 # Tournament testing script
│── App.jsx                  # ReactJS board UI
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- Node.js & npm
- A Google AI Studio API key (for Gemini integration)

### Backend

```bash
# Install Python dependencies
pip install flask numpy google-generativeai

# Add your Gemini API key to the backend
# In app.py or a .env file, set:
API_KEY = "your-google-ai-studio-key"

# Run the Flask server
python app.py
```

### Frontend

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the dev server
npm start
```

The app will be available at `http://localhost:3000`, with the Flask API running on `http://localhost:5000`.

---

## AI Agents

### Evaluation Heuristic
All search-based agents share a **4×4 sliding window** heuristic that scores board states based on:
- Control of the center 4 columns
- Consecutive same-colour pieces in any direction (horizontal, vertical, diagonal)

### Minimax
Recursively explores the game tree, assuming optimal play from both sides. Depth is capped at **4–8 moves** to keep computation feasible on a standard 6×7 board.

### Alpha-Beta Pruning
An optimized Minimax that prunes branches guaranteed not to affect the final decision. In testing, it reduced nodes explored by **65.5%** and ran approximately **5× faster** than plain Minimax.

### ExpectiMinimax
Extends Minimax with **chance nodes** to handle stochastic elements. In non-random mode, it models the opponent as playing with uniform probability over valid moves. In stochastic mode, it accounts for random piece drops (25% probability) by weighting outcomes accordingly.

### Gemini (LLM Agent)
Uses the Google Gemini 2.5 Pro model via API. The backend formats the current board state into a prompt and asks Gemini to return the best column number. Useful for benchmarking classical algorithms against an LLM. Falls back to a random move on API errors.

---

## State Space

| Mode | State Space Complexity |
|------|------------------------|
| Standard | O(3^(n×m)) |
| Stochastic | O(3^(n×m)) (random pieces abstracted as a 4th rare state) |

For a standard 6×7 board: ~4.5 × 10¹² possible states. A **depth limit of 4** is applied to keep computation feasible.

---

## Testing & Results

Tournament testing was run with 3-trial and 100-trial configurations, with and without Gemini, across standard and stochastic modes.

### Win Rates (No Gemini, 100 trials)

| Agent | Win Rate |
|-------|----------|
| Minimax | 62.5% |
| Alpha-Beta | 62.5% |
| ExpectiMinimax | 62.5% |
| Random | 12.5% |

### Win Rates (With Gemini, 2 trials)

| Agent | Win Rate |
|-------|----------|
| Minimax | 70% |
| Alpha-Beta | 70% |
| ExpectiMinimax | 70% |
| Gemini | 30% |
| Random | 10% |

### Node Efficiency (6×7 board, depth 4)

| Agent | Nodes Evaluated | Nodes/Second |
|-------|----------------|--------------|
| Minimax | 2,800 | ~9,783 |
| Alpha-Beta | 965 | ~10,781 |
| ExpectiMinimax | 2,800 | ~10,024 |

### Stochastic Mode Impact
With 25% random piece drop probability, the Random agent's win rate increased slightly (+4%), while ExpectiMinimax dropped slightly (−4%), reflecting the increased unpredictability.

---

## Known Limitations

- **Gemini API errors:** HTTP 503 (model overloaded) and 429 (rate limit exceeded) reduce trial completeness. On error, the system defaults to a random move. A future improvement would be falling back to a classical algorithm instead.
- **Depth limit:** Set to 4 to avoid the full ~4.5 × 10¹² state space; deeper search would improve AI quality at the cost of speed.
- **First-mover advantage:** The first agent to move has a slight structural advantage in Connect Four; a turn-alternating system was implemented to reduce this bias.

---

## References

1. Science Buddies. (2024, February 7). *Simple Explanation of the Minimax Algorithm with Alpha-Beta Pruning with Connect 4* [Video]. YouTube. https://www.youtube.com/watch?v=DV5d31z1xTI&t=197s

2. Dartmouth College. (2016). *Lab Assignment 6: Connect Four*. cs.dartmouth.edu. https://www.cs.dartmouth.edu/~scot/cs10/lab/lab6/lab6.html
