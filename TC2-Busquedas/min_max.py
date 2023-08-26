# Min max algorithm for TIC TAC TOE!!!

from math import inf as infinity
import random  # Import random module for handling randomness

# Initialize the game board as a 3x3 grid
board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

PC = +1  # Constant representing the computer's player
HUMAN = -1  # Constant representing the human player


def evaluate(state):
    """
    Evaluate the current state of the game.

    Parameters:
        state (list[list[int]]): The current game board state.

    Returns:
        int: +1 if the computer wins, -1 if the human wins, or 0 for a draw.
    """
    if (gameover(state, PC)):
        score = +1
    elif (gameover(state, HUMAN)):
        score = -1
    else:
        score = 0
    return score


def gameovergame(state):
    """
    Check if the game is over for both players.

    Parameters:
        state (list[list[int]]): The current game board state.

    Returns:
        bool: True if the game is over for both players, False otherwise.
    """
    return gameover(state, HUMAN) or gameover(state, PC)


def gameover(state, player):
    """
    Check if the game is over for a specific player.

    Parameters:
        state (list[list[int]]): The current game board state.
        player (int): The player to check for (PC or HUMAN).

    Returns:
        bool: True if the specified player has won, False otherwise.
    """
    winstate = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        # Fix the error in the last combination
        [state[2][0], state[2][1], state[2][2]]
    ]
    if [player, player, player] in winstate:
        return True
    else:
        return False


def domove(i, j):
    """
    Make a move on the game board for the human player if it's a valid move.

    Parameters:
        i (int): Row index of the move.
        j (int): Column index of the move.
    """
    if validmove(i, j):
        board[i][j] = HUMAN


def validmove(i, j):
    """
    Check if a move at the given position (i, j) is valid.

    Parameters:
        i (int): Row index of the move.
        j (int): Column index of the move.

    Returns:
        bool: True if the move is valid, False otherwise.
    """
    if [i, j] in clear_cells(board):
        return True
    else:
        return False


def clear_cells(state):
    """
    Get a list of empty cells on the game board.

    Parameters:
        state (list[list[int]]): The current game board state.

    Returns:
        list[list[int]]: A list of coordinates of empty cells on the game board.
    """
    cells = []
    for i, row in enumerate(state):
        for j, cell in enumerate(row):
            if cell == 0:
                cells.append([i, j])
    return cells


def minimax(state, depth, player):
    """
    Implement the Minimax algorithm to find the best move for a player.

    Parameters:
        state (list[list[int]]): The current game board state.
        depth (int): The current depth in the search tree.
        player (int): The player for whom to find the best move.

    Returns:
        list[int]: A list containing the best move's coordinates (i, j) and its score.
    """
    if player == PC:
        best = [-1, -1, -infinity]
    else:
        best = [-1, -1, +infinity]

    # Base case: If the maximum depth is reached or the game is over, evaluate the state.
    if depth == 0 or gameovergame(state):
        score = evaluate(state)
        return [-1, -1, score]

    # Iterate through each available cell on the board.
    for cell in clear_cells(state):
        i, j = cell[0], cell[1]

        # Simulate making a move for the current player and proceed with Minimax recursion.
        state[i][j] = player
        score = minimax(state, depth - 1, -player)
        # Reset the cell to its original state after simulating the move.
        state[i][j] = 0

        # Update the move's coordinates based on current iteration.
        score[0], score[1] = i, j

        # If the current player is the computer, maximize the score.
        if player == PC:
            if score[2] > best[2]:
                best = score
        # If the current player is the human, minimize the score.
        else:
            if score[2] < best[2]:
                best = score

    return best


def computer_move():
    """
    Make a move for the computer using the Minimax algorithm and handle randomness.

    Returns:
        tuple[int, int]: The row and column indices of the computer's move.
    """
    depth = len(clear_cells(board))
    if depth == 0 or gameovergame(board):
        return -1, -1

    if depth == 9:  # If it's the first move, play in the center
        return 1, 1

    best_move = minimax(board, depth, PC)

    # Gather all the best moves with the same score
    best_moves = []
    for cell in clear_cells(board):
        i, j = cell[0], cell[1]
        if board[i][j] == 0:
            board[i][j] = PC
            move_score = minimax(board, depth - 1, HUMAN)[2]
            board[i][j] = 0
            if (PC == 1 and move_score >= best_move[2]) or (PC == -1 and move_score <= best_move[2]):
                best_moves.append((i, j))

    return random.choice(best_moves)  # Choose a random move from the best ones


def print_board(state):
    symbols = {0: " ", 1: "X", -1: "O"}
    for row in state:
        print(" | ".join(symbols[cell] for cell in row))
        print("-" * 9)


# Loop for player vs computer game
while len(clear_cells(board)) > 0 and not gameovergame(board):
    print("Current Board:")
    print_board(board)

    # Human's turn
    while True:
        row = int(input("Enter the row (0, 1, or 2) for your move: "))
        col = int(input("Enter the column (0, 1, or 2) for your move: "))
        if validmove(row, col):
            domove(row, col)
            break
        else:
            print("Invalid move! Try again.")

    if len(clear_cells(board)) == 0 or gameovergame(board):
        break

    # Computer's turn
    row, col = computer_move()
    domove(row, col)

    if len(clear_cells(board)) == 0 or gameovergame(board):
        break

print("Final Board:")
print_board(board)

if gameover(board, PC):
    print("Computer wins!")
elif gameover(board, HUMAN):
    print("You win!")
else:
    print("It's a draw!")
