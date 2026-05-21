import connect4game
import algorithms as algo
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#program runs 2 games per matchup and outputs all relevant testing data and graphs. will run gemini

def random_move(board, depth=4):
    """Random move selection"""
    return connect4game.random_valid_column(board)

def gemini_move(board, depth=4):
    """Wrapper for Gemini AI that handles errors"""
    try:
        col = connect4game.gemini_prompt(board)
        return col
    except Exception as e:
        print(f"Gemini error: {e}, using random move")
        return connect4game.random_valid_column(board)
    
# Algorithm dictionary
algorithms_dict = {
    'Random': random_move,
    'Minimax': algo.minimax_move,
    'AlphaBeta': algo.alpha_beta_move,
    'ExpectiMinimax': algo.expectiminimax_move,
    'Gemini': gemini_move, #if program takes too long then comment out gemini to find data pertaining to algorithms
}

def simulate_game(move_fn_a, move_fn_b, depth_a=4, depth_b=4, max_moves=100):
    """
    Simulate a game between two AI agents
    Returns 'A' if first agent wins, 'B' if second agent wins, 'Draw' if tie
    """
    board = connect4game.createBoard()
    turn = 0  # 0 = Agent A's turn, 1 = Agent B's turn
    
    for x in range(max_moves):
        # Determine which agent and piece to use
        current_fn = move_fn_a if turn == 0 else move_fn_b
        current_depth = depth_a if turn == 0 else depth_b
        current_piece = connect4game.PLAYER_PIECE if turn == 0 else connect4game.AI_PIECE
        
        try:
            # Get move from current agent
            col = current_fn(board, depth=current_depth)
            
            # Validate move
            if not connect4game.isValidMove(board, col):
                # Invalid move = automatic loss
                return "B" if turn == 0 else "A"
            
            # Make the move
            row = connect4game.getNextRow(board, col)
            connect4game.playMove(board, row, col, current_piece)
            
            # Check for win
            if connect4game.checkWinningMove(board, current_piece):
                return "A" if turn == 0 else "B"
            
            # Check for draw (board full)
            if not any(connect4game.EMPTY in row for row in board):
                return "Draw"
            
            # Switch turns
            turn = 1 - turn
            
        except Exception as e:
            # If agent crashes, it loses
            print(f"Agent error: {e}")
            return "B" if turn == 0 else "A"
    
    # If max moves reached, it's a draw
    return "Draw"


def run_tournament(trials=3, depth=4, verbose=True):
    """Run a tournament between all algorithms"""
    results = {}
    total_matches = len(algorithms_dict) * len(algorithms_dict)
    current_match = 0
    
    print(f"Running tournament with {trials} trials per matchup...\n")
    
    # Run tournament
    for name_a, fn_a in algorithms_dict.items():
        for name_b, fn_b in algorithms_dict.items():
            current_match += 1
            key = f"{name_a} vs {name_b}"
            results[key] = {'A_wins': 0, 'B_wins': 0, 'Draws': 0}
            
            if verbose:
                print(f"Match {current_match}/{total_matches}: {key}")
            
            # Run multiple trials with alternating starting positions
            for trial in range(trials):
                if verbose and trials > 1:
                    print(f"  Trial {trial + 1}/{trials}", end="")
                
                # Alternate who goes first each trial
                if trial % 2 == 0:
                    # Even trials: A goes first
                    result = simulate_game(fn_a, fn_b, depth_a=depth, depth_b=depth)
                else:
                    # Odd trials: B goes first (swap A and B)
                    result = simulate_game(fn_b, fn_a, depth_a=depth, depth_b=depth)
                    # Need to flip the result since we swapped the players
                    if result == 'A':
                        result = 'B'
                    elif result == 'B':
                        result = 'A'
                    # 'Draw' stays the same
                
                if result == 'A':
                    results[key]['A_wins'] += 1
                elif result == 'B':
                    results[key]['B_wins'] += 1
                else:
                    results[key]['Draws'] += 1
                
                if verbose and trials > 1:
                    starter = name_a if trial % 2 == 0 else name_b
                    print(f" -> {result} ({starter} started)")
            
            if verbose:
                wins_a = results[key]['A_wins']
                wins_b = results[key]['B_wins']
                draws = results[key]['Draws']
                print(f"  Result: {name_a}: {wins_a}, {name_b}: {wins_b}, Draws: {draws}\n")
    
    return results

def print_tournament_summary(results):
    """Print a summary of tournament results"""
    print("=" * 60)
    print("TOURNAMENT RESULTS SUMMARY")
    print("=" * 60)
    
    # Calculate overall statistics
    algorithm_stats = {}
    for name in algorithms_dict.keys():
        algorithm_stats[name] = {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0}
    
    # Aggregate results
    for match, outcome in results.items():
        name_a, name_b = match.split(' vs ')
        
        # Update stats for algorithm A
        algorithm_stats[name_a]['wins'] += outcome['A_wins']
        algorithm_stats[name_a]['losses'] += outcome['B_wins']
        algorithm_stats[name_a]['draws'] += outcome['Draws']
        algorithm_stats[name_a]['games'] += sum(outcome.values())
        
        # Update stats for algorithm B
        algorithm_stats[name_b]['wins'] += outcome['B_wins']
        algorithm_stats[name_b]['losses'] += outcome['A_wins']
        algorithm_stats[name_b]['draws'] += outcome['Draws']
        algorithm_stats[name_b]['games'] += sum(outcome.values())
    
    # Print individual match results
    print("\nDETAILED MATCH RESULTS:")
    print("-" * 60)
    for match, outcome in results.items():
        name_a, name_b = match.split(' vs ')
        print(f"{name_a:15} vs {name_b:15} | {name_a}: {outcome['A_wins']:2}, {name_b}: {outcome['B_wins']:2}, Draws: {outcome['Draws']:2}")
    
    # Print overall statistics
    print(f"\nOVERALL STATISTICS:")
    print("-" * 60)
    print(f"{'Algorithm':<15} {'Wins':<6} {'Losses':<7} {'Draws':<6} {'Win Rate':<8} {'Games':<6}")
    print("-" * 60)
    
    for name, stats in sorted(algorithm_stats.items(), key=lambda x: x[1]['wins'], reverse=True):
        wins = stats['wins']
        losses = stats['losses']
        draws = stats['draws']
        games = stats['games']
        win_rate = (wins / games * 100) if games > 0 else 0
        
        print(f"{name:<15} {wins:<6} {losses:<7} {draws:<6} {win_rate:<7.1f}% {games:<6}")
    
    return algorithm_stats


def scalability_test(algorithms, depth=4, trials=3):
    """Test algorithm performance on different board sizes"""    
    board_sizes = [(4, 5), (6, 7), (8, 9)]
    scalability_results = {}
    
    print("\n" + "=" * 60)
    print("SCALABILITY TESTING")
    print("=" * 60)
    
    for rows, cols in board_sizes:
        print(f"\nTesting on {rows}x{cols} board:")
        print("-" * 40)
        
        size_results = {}
        
        for alg_name, alg_func in algorithms.items():
            # Skip random algorithm for scalability testing
            if alg_name == 'Random':
                continue
                
            total_time = 0
            total_nodes = 0
            
            for trial in range(trials):
                # Create a deterministic board state instead of random
                connect4game.NUM_ROW = rows
                connect4game.NUM_COL = cols
                board = connect4game.createBoard(rows, cols)
                
                # Add some deterministic pieces to create a mid-game state
                moves_to_make = min(3, rows - 1)
                for i in range(moves_to_make):
                    col = i % cols  # Deterministic column selection
                    if connect4game.isValidMove(board, col):
                        row = connect4game.getNextRow(board, col)
                        # Alternate pieces deterministically
                        piece = connect4game.PLAYER_PIECE if i % 2 == 0 else connect4game.AI_PIECE
                        connect4game.playMove(board, row, col, piece)
                
                # Measure algorithm performance
                algo.node_count = 0  # Reset counter
                start_time = time.time()
                
                try:
                    move = alg_func(board, depth)
                    end_time = time.time()
                    nodes_evaluated = algo.node_count
                    
                    total_time += (end_time - start_time)
                    total_nodes += nodes_evaluated

                        
                except Exception as e:
                    end_time = time.time()
                    total_time += (end_time - start_time)
                    print(f"    Error in {alg_name} trial {trial + 1}: {e}")
            
            # Calculate averages
            avg_time = total_time / trials if trials > 0 else 0
            avg_nodes = total_nodes // trials if trials > 0 else 0
            
            size_results[alg_name] = {
                'avg_time': avg_time,
                'avg_nodes': avg_nodes,
            }
            
            print(f"  {alg_name:<15}: {avg_time:.4f}s, {avg_nodes:>6} nodes")
        
        scalability_results[f"{rows}x{cols}"] = size_results
    
    #restore original
    connect4game.NUM_ROW = 6
    connect4game.NUM_COL = 7
    return scalability_results


def create_performance_graphs(tournament_results, scalability_results, node_performance):
    """Create comprehensive performance analysis graphs"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Extract data from tournament results
    algorithm_stats = {}
    for name in algorithms_dict.keys():
        algorithm_stats[name] = {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0}
    
    # Aggregate tournament results
    for match, outcome in tournament_results.items():
        name_a, name_b = match.split(' vs ')
        
        algorithm_stats[name_a]['wins'] += outcome['A_wins']
        algorithm_stats[name_a]['losses'] += outcome['B_wins']
        algorithm_stats[name_a]['draws'] += outcome['Draws']
        algorithm_stats[name_a]['games'] += sum(outcome.values())
        
        algorithm_stats[name_b]['wins'] += outcome['B_wins']
        algorithm_stats[name_b]['losses'] += outcome['A_wins']
        algorithm_stats[name_b]['draws'] += outcome['Draws']
        algorithm_stats[name_b]['games'] += sum(outcome.values())
    
    # Calculate win rates
    algorithms = list(algorithms_dict.keys())
    win_rates = []
    for alg in algorithms:
        win_rate = (algorithm_stats[alg]['wins'] / algorithm_stats[alg]['games'] * 100) if algorithm_stats[alg]['games'] > 0 else 0
        win_rates.append(win_rate)
    
    # Extract scalability data
    board_sizes = ['4x5', '6x7', '8x9']
    execution_times = {}
    nodes_expanded_scalability = {}
    
    for alg in ['Minimax', 'AlphaBeta', 'ExpectiMinimax', 'Gemini']:
        execution_times[alg] = []
        nodes_expanded_scalability[alg] = []
        
        for size in board_sizes:
            if size in scalability_results and alg in scalability_results[size]:
                execution_times[alg].append(scalability_results[size][alg]['avg_time'])
                nodes_expanded_scalability[alg].append(scalability_results[size][alg]['avg_nodes'])
            else:
                execution_times[alg].append(0)
                nodes_expanded_scalability[alg].append(0)
    
    # Create head-to-head win matrix
    alg_list = list(algorithms_dict.keys())
    win_matrix = np.zeros((len(alg_list), len(alg_list)))
    
    for match, outcome in tournament_results.items():
        name_a, name_b = match.split(' vs ')
        i = alg_list.index(name_a)
        j = alg_list.index(name_b)
        total_games = sum(outcome.values())
        win_matrix[i][j] = (outcome['A_wins'] / total_games * 100) if total_games > 0 else 0
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Win Rate Comparison
    ax1 = plt.subplot(2, 3, 1)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars1 = plt.bar(algorithms, win_rates, color=colors[:len(algorithms)])
    plt.title('Algorithm Win Rate Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylim(0, max(win_rates) * 1.1)
    
    # Add value labels on bars
    for bar, rate in zip(bars1, win_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(win_rates)*0.02,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Node Expansion Comparison (Single Move)
    ax2 = plt.subplot(2, 3, 2)
    node_algs = list(node_performance.keys())
    node_counts = list(node_performance.values())
    bars2 = plt.bar(node_algs, node_counts, color=['#4ECDC4', '#45B7D1', '#96CEB4'])
    plt.title('Node Expansion (Single Move, Depth 4)', fontsize=14, fontweight='bold')
    plt.ylabel('Nodes Expanded', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    
    # Add value labels
    for bar, count in zip(bars2, node_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(node_counts)*0.02,
                 f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Execution Time Scalability
    ax3 = plt.subplot(2, 3, 3)
    x_pos = np.arange(len(board_sizes))
    width = 0.25
    
    for i, (alg, times) in enumerate(execution_times.items()):
        offset = (i - 1) * width
        plt.bar(x_pos + offset, times, width, label=alg, alpha=0.8)
    
    plt.title('Execution Time Scalability', fontsize=14, fontweight='bold')
    plt.xlabel('Board Size', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.xticks(x_pos, board_sizes)
    plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    
    # Nodes Expanded Scalability
    ax4 = plt.subplot(2, 3, 4)
    for alg, nodes in nodes_expanded_scalability.items():
        if any(n > 0 for n in nodes):  # Only plot if we have data
            plt.plot(board_sizes, nodes, marker='o', linewidth=2, markersize=8, label=alg)
    
    plt.title('Node Expansion Scalability', fontsize=14, fontweight='bold')
    plt.xlabel('Board Size', fontsize=12)
    plt.ylabel('Nodes Expanded', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    
    # Algorithm Efficiency Comparison
    ax5 = plt.subplot(2, 3, 5)
    # Calculate efficiency as nodes per second for each algorithm on 6x7 board (index 1), 
    # this will make it seem that AB goes through more nodes per second but all it is stating is it can travserse more nodes faster in a second when in relaity it lates little time and traverses though a small amount of nodes
    efficiency_data = {}
    for alg in ['Minimax', 'AlphaBeta', 'ExpectiMinimax', 'Gemini']:
        if (len(nodes_expanded_scalability[alg]) > 1 and len(execution_times[alg]) > 1 and 
            execution_times[alg][1] > 0):
            nodes_per_sec = nodes_expanded_scalability[alg][1] / execution_times[alg][1]
            efficiency_data[alg] = nodes_per_sec
    
    if efficiency_data:
        algs_eff = list(efficiency_data.keys())
        efficiency_values = list(efficiency_data.values())
        bars5 = plt.bar(algs_eff, efficiency_values, color=['#4ECDC4', '#45B7D1', '#96CEB4'])
        
        plt.title('Algorithm Efficiency\n(Nodes/Second on 6x7 Board)', fontsize=14, fontweight='bold')
        plt.ylabel('Nodes per Second', fontsize=12)
        plt.xlabel('Algorithm', fontsize=12)
        
        # Add value labels
        for bar, eff in zip(bars5, efficiency_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(efficiency_values)*0.02,
                     f'{eff:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
    
    # Adjust layout and display
    plt.tight_layout(pad=3.0)
    plt.suptitle('Complete Connect 4 AI Algorithm Performance Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Save the figure
    plt.savefig('complete_connect4_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Performance analysis graphs saved as 'complete_connect4_performance_analysis.png'")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("ALGORITHM PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Print efficiency comparison if we have node performance data
    if 'Minimax' in node_performance and 'AlphaBeta' in node_performance:
        minimax_nodes = node_performance['Minimax']
        alphabeta_nodes = node_performance['AlphaBeta']
        reduction = ((minimax_nodes - alphabeta_nodes) / minimax_nodes * 100)
        print(f"\nAlpha-Beta Pruning Efficiency:")
        print("-" * 60)
        print(f"- Reduces nodes by {reduction:.1f}% compared to Minimax")
        
        if ('Minimax' in execution_times and 'AlphaBeta' in execution_times and 
            len(execution_times['Minimax']) > 1 and len(execution_times['AlphaBeta']) > 1 and
            execution_times['AlphaBeta'][1] > 0):
            speedup = execution_times['Minimax'][1] / execution_times['AlphaBeta'][1]
            print(f"- Achieves {speedup:.1f}x speedup on 6x7 board")

def writeToOutput(best_col,numTurns):
    #writes to the move.txt file what we are about to do
    #call this inside some sort of while loop that checks if the game is not over
    with open('move.txt','w') as file:
        file.write(f"For move #{numTurns} I will drop my piece in column #{best_col}")

if __name__ == '__main__':    
    # Run tournament
    TRIALS = 2  # Number of games per matchup
    DEPTH = 4   # Search depth for algorithms
    
    tournament_results = run_tournament(trials=TRIALS, depth=DEPTH, verbose=True)
    algorithm_stats = print_tournament_summary(tournament_results)
    
    # Additional performance testing
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTING")
    print("=" * 60)
    
    # Test node counts for deterministic algorithms only
    test_board = connect4game.createBoard()
    print("\nNode count comparison (single move at depth 4):")
    deterministic_algorithms = ['Minimax', 'AlphaBeta', 'ExpectiMinimax']
    node_performance = {}
    
    for name, fn in algorithms_dict.items():
        if name in deterministic_algorithms:
            algo.node_count = 0  # Reset counter
            fn(test_board, depth=4)
            node_performance[name] = algo.node_count
            print(f"{name:<15}: {algo.node_count:>8} nodes")

    # Run scalability test
    scalability_results = scalability_test(algorithms_dict, trials=3)    
    # Create performance graphs
    create_performance_graphs(tournament_results, scalability_results, node_performance)


