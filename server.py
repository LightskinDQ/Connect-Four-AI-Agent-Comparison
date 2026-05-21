import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from algorithms import *
import connect4game
import randomConnectFour

app = Flask('ConnectFourGame')
CORS(app)

@app.route("/", methods=["POST"])
def ai_move():
    data = request.get_json()
    board = np.array(data.get("board"))
    
    connect4game.NUM_ROW = len(board)
    connect4game.NUM_COL = len(board[0]) if board.size > 0 else 0
    
    algo = data.get("algo", "alpha-beta")#default algorithm

    if algo.lower() == "minimax":
        move = minimax_move(board)
    elif algo.lower() == "expectiminimax":
        move = expectiminimax_random_move(board)
    elif algo.lower() == "gemini":
        move = connect4game.gemini_prompt(board)
    elif algo.lower() == "gemini(random)":
        move = randomConnectFour.gemini_prompt(board)
    else: 
        move = alpha_beta_move(board)

    return jsonify({"move": int(move)})
