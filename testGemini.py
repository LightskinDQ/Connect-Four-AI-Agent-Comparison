import connect4game
import algorithms

def simulate_game(move_fn_a, move_fn_b, depth_a=4, depth_b=4, max_moves=100):
    board = connect4game.createBoard()
    turn = 0  # 0 = Agent A's turn, 1 = Agent B's turn

    for _ in range(max_moves):
        current_fn = move_fn_a if turn == 0 else move_fn_b
        current_depth = depth_a if turn == 0 else depth_b
        current_piece = connect4game.PLAYER_PIECE if turn == 0 else connect4game.AI_PIECE

        try:
            col = current_fn(board, depth=current_depth)

            if not connect4game.isValidMove(board, col):
                return "B" if turn == 0 else "A"

            row = connect4game.getNextRow(board, col)
            connect4game.playMove(board, row, col, current_piece)

            if connect4game.checkWinningMove(board, current_piece):
                return "A" if turn == 0 else "B"

            if not any(connect4game.EMPTY in row for row in board):
                return "Draw"

            turn = 1 - turn

        except Exception as e:
            print(f"Agent error: {e}")
            return "B" if turn == 0 else "A"

    return "Draw"

def gemini_move(board, depth=4):
    try:
        return connect4game.gemini_prompt(board)
    except Exception as e:
        print(f"Gemini error: {e}, using random move")
        return connect4game.random_valid_column(board)

algorithms_to_test = {
    'Minimax': algorithms.minimax_move,
    'AlphaBeta': algorithms.alpha_beta_move,
    'ExpectiMinimax': algorithms.expectiminimax_move,
}

def run_vs_gemini(trials=10, depth=4, verbose=True):
    results = {}

    for name, algo_fn in algorithms_to_test.items():
        wins, losses, draws = 0, 0, 0
        print(f"\n{name} vs Gemini:")

        for i in range(trials):
            result = simulate_game(algo_fn, gemini_move, depth_a=depth, depth_b=depth)
            if result == "A":
                wins += 1
            elif result == "B":
                losses += 1
            else:
                draws += 1

            if verbose:
                print(f"  Game {i+1}/{trials} -> {result}")

        results[name] = {'Wins': wins, 'Losses': losses, 'Draws': draws}

    return results

def print_vs_gemini_summary(results):
    print("\n===== SUMMARY: AI vs Gemini =====")
    print(f"{'Algorithm':<15} {'Wins':<6} {'Losses':<7} {'Draws':<6} {'Win Rate (%)':<12}")
    for name, res in results.items():
        total = res['Wins'] + res['Losses'] + res['Draws']
        win_rate = 100 * res['Wins'] / total if total > 0 else 0
        print(f"{name:<15} {res['Wins']:<6} {res['Losses']:<7} {res['Draws']:<6} {win_rate:<12.1f}")

if __name__ == '__main__':
    print("Testing algorithms vs Gemini...\n")
    results = run_vs_gemini(trials=3, depth=3, verbose=True)
    print_vs_gemini_summary(results)
