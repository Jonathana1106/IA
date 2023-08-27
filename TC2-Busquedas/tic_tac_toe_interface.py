# Min max algorithm for TIC TAC TOE!!!


import platform
import time
import os
import random
from math import inf as infinity
from random import choice


PC = +1  # Constant representing the computer's player
PC2 = -1  # Constant representing the computer's 2 player


def evaluate(state):
    """
    Evaluate the current state of the game.

    Parameters:
        state (list[list[int]]): The current game board state.

    Returns:
        int: +1 if the computer wins, -1 if the computer 2 wins, or 0 for a draw.
    """
    if (gameover(state, PC)):
        score = +1
    elif (gameover(state, PC2)):
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
    return gameover(state, PC2) or gameover(state, PC)


def gameover(state, player):
    """
    Check if the game is over for a specific player.

    Parameters:
        state (list[list[int]]): The current game board state.
        player (int): The player to check for (PC or PC2).

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
        [state[2][0], state[1][1], state[0][2]]
    ]
    if ([player, player, player] in winstate):
        return True
    else:
        return False


def domove(i, j, player):
    """
    Make a move on the game board for the computer 2 player if it's a valid move.

    Parameters:
        i (int): Row index of the move.
        j (int): Column index of the move.
    """
    if validmove(i, j):
        board[i][j] = player
        return True
    else:
        return False


def validmove(i, j):
    """
    Check if a move at the given position (i, j) is valid.

    Parameters:
        i (int): Row index of the move.
        j (int): Column index of the move.

    Returns:
        bool: True if the move is valid, False otherwise.
    """
    if [i, j] in clearcells(board):
        return True
    else:
        return False


def clearcells(state):
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

    best_moves = []
    # Iterate through each available cell on the board.
    for cell in clearcells(state):
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
                best_moves = [score]
            elif score[2] == best[2]:
                best_moves.append(score)
        # If the current player is the computer 2, minimize the score.
        else:
            if score[2] < best[2]:
                best = score
                best_moves = [score]
            elif score[2] == best[2]:
                best_moves.append(score)

    # Choose a random move from the list of best moves
    return random.choice(best_moves)


def mainIAvsIA():
    """
    Main function to run the Tic Tac Toe game between two AI players (IA vs IA).

    This function initializes the game and starts the game loop where two AI players play against each other.
    """
    initialize_board()  # Reset the board at the beginning of each game
    p1_choice, p2_choice = 'X', 'O'  # Set symbols for the players
    print("IA vs IA:")
    while len(clearcells(board)) > 0 and not gameovergame(board):
        iaturn(p1_choice, p2_choice, 1)  # IA 1 turn
        if not gameovergame(board):  # Check if the game is still ongoing
            iaturn(p1_choice, p2_choice, 2)  # IA 2 turn

    # Return the winner or "Draw"
    winner = announce_winner(p1_choice, p2_choice)
    print("IA vs IA Winner:", winner)
    return winner


def mainIAvsRand():
    """
    Main function to run the Tic Tac Toe game between AI and Random.

    This function initializes the game and starts the game loop where an AI player plays against Random.
    """
    initialize_board()  # Reset the board at the beginning of each game
    p1_choice, p2_choice = 'X', 'O'  # Set symbols for the players
    print("IA vs Random:")
    while len(clearcells(board)) > 0 and not gameovergame(board):
        iaturn(p1_choice, p2_choice, 1)  # IA MINMAX  turn
        if not gameovergame(board):  # Check if the game is still ongoing
            randomTurn(p1_choice, p2_choice)  # IA Random's turn

    # Return the winner or "Draw"
    winner = announce_winner(p1_choice, p2_choice)
    print("IA vs Rand Winner:", winner)
    return winner


def mainRandvsIA():
    """
    Main function to run the Tic Tac Toe game between Random and AI.

    This function initializes the game and starts the game loop where Random plays against an AI player.
    """
    initialize_board()  # Reset the board at the beginning of each game
    p1_choice, p2_choice = 'X', 'O'  # Set symbols for the players
    print("Random vs IA:")
    while len(clearcells(board)) > 0 and not gameovergame(board):
        randomTurn(p1_choice, p2_choice)  # IA Random's turn
        if not gameovergame(board):  # Check if the game is still ongoing
            iaturn(p1_choice, p2_choice, 1)  # IA MINMAX turn

    # Return the winner or "Draw"
    winner = announce_winner(p1_choice, p2_choice)
    print("Rand vs IA Winner:", winner)
    return winner

# Initialize the game board as a 3x3 grid
def initialize_board():
    """
    Initializes the game board with empty cells.
    """
    global board
    board = [[0 for _ in range(3)] for _ in range(3)]


def render(state, pchoice, hchoice):
    """
    Renders the current state of the game board on the console.

    Parameters:
        state (list[list[int]]): The current game board state.
        pchoice (str): The computer's choice ('X' or 'O').
        hchoice (str): The computer's 2 choice ('X' or 'O').
    """
    chars = {
        -1: hchoice,
        +1: pchoice,
        0: ' '
    }
    strline = '---------------'

    print('\n' + strline)
    for row in state:
        for cell in row:
            symbol = chars[cell]
            print(f'| {symbol} |', end='')
        print('\n' + strline)


def announce_winner(p1_choice, p2_choice):
    """
    Announce the winner of the game or declare a draw.

    Parameters:
        p1_choice (str): The symbol for the first player ('X' or 'O').
        p2_choice (str): The symbol for the second player ('X' or 'O').

    Returns:
        int: 1 if IA Player 1 wins, 2 if IA Player 2 wins, or 0 for a draw.
    """
    if gameover(board, PC):
        print('PLAYER IA MINMAX WINS!')
        return 1
    elif gameover(board, PC2):
        print('PLAYER IA Random WINS!')
        return 2
    else:
        print('DRAW!')
        return 0


def iaturn(p1_choice, p2_choice, currentPlayer):
    """
    Executes the AI player's turn (Minimax) in the game.

    Parameters:
        p1_choice (str): The symbol for the first player ('X' or 'O').
        p2_choice (str): The symbol for the second player ('X' or 'O').
    """
    depth = len(clearcells(board))
    if depth == 0 or gameovergame(board):
        return

    if (currentPlayer == 1):
        player_symbol = p1_choice
        currentPC = PC
    else:
        player_symbol = p2_choice
        currentPC = PC2

    print(f'Player IA MINMAX turn [{player_symbol}]')

    # PC is the player index for Player 1
    move = minimax(board, depth, currentPC)
    x, y = move[0], move[1]
    domove(x, y, currentPC)
    render(board, p1_choice, p2_choice)
    time.sleep(1)


def randomTurn(p1_choice, p2_choice):
    """
    Executes the AI player's turn (Random) in the game.

    Parameters:
        p1_choice (str): The symbol for the first player ('X' or 'O').
        p2_choice (str): The symbol for the second player ('X' or 'O').
    """
    depth = len(clearcells(board))
    if depth == 0 or gameovergame(board):
        return

    player_symbol = p2_choice
    print(f'Player IA RANDOM turn [{player_symbol}]')
    legal_moves = clearcells(board)
    random_move = random.choice(legal_moves)
    x, y = random_move[0], random_move[1]
    domove(x, y, PC2)  # Note: PC2 is used here to represent Player 2 (Random)
    render(board, p1_choice, p2_choice)
    time.sleep(1)


def main():
    """
    Main function to run a series of Tic Tac Toe games and track the results.

    This function runs multiple Tic Tac Toe games between different combinations of AI players and
    tracks the number of wins, losses, and draws for each scenario.

    Prints the results for each scenario at the end of all games.
    """
    IAvsIA = [0, 0, 0]   # Tracks wins, losses, and draws for AI vs AI games
    IAvsRand = [0, 0, 0] # Tracks wins, losses, and draws for AI vs Random games
    RandvsIA = [0, 0, 0] # Tracks wins, losses, and draws for Random vs AI games

    for _ in range(5):  # Play 5 games
        winner_IAvsRand = mainIAvsRand()
        if winner_IAvsRand == 1:
            IAvsRand[0] += 1
        elif winner_IAvsRand == -1:
            IAvsRand[1] += 1
        else:
            IAvsRand[2] += 1

        winner_RandvsIA = mainRandvsIA()
        if winner_RandvsIA == -1:
            RandvsIA[0] += 1
        elif winner_RandvsIA == 1:
            RandvsIA[1] += 1
        else:
            RandvsIA[2] += 1

        winner_IAvsIA = mainIAvsIA()
        if winner_IAvsIA == 1:
            IAvsIA[0] += 1
        elif winner_IAvsIA == -1:
            IAvsIA[1] += 1
        else:
            IAvsIA[2] += 1

    print("IA vs Random:")
    print_results(IAvsRand)

    print("Random vs IA:")
    print_results(RandvsIA)

    print("IA vs IA:")
    print_results(IAvsIA)


def print_results(results):
    """
    Print the results of the games (wins, losses, and draws) for a specific scenario.

    Parameters:
        results (list[int]): A list containing the counts of wins, losses, and draws.
    """
    labels = ["Gano", "Perdio", "Empato"]

    for label, count in zip(labels, results):
        print(f"{label}: {count}")

    print()


if __name__ == '__main__':
    main()
