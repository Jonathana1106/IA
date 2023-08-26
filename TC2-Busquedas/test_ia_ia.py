# Min max algorithm for TIC TAC TOE!!!


import platform
import time
import os
import random
from math import inf as infinity
from random import choice


# Initialize the game board as a 3x3 grid
board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

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


def main():
    """
    Main function to run the Tic Tac Toe game.

    This function initializes the game and starts the game loop where two AI players play against each other.
    """
    p1_choice, p2_choice = 'X', 'O'  # Set symbols for the players

    while len(clearcells(board)) > 0 and not gameovergame(board):
        iaturn1(p1_choice, p2_choice)
        if not gameovergame(board):  # Check if the game is still ongoing
            iaturn2(p1_choice, p2_choice)  # IA Random's turn

    announce_winner(p1_choice, p2_choice)


def choose_symbols():
    """
    Prompt the player to choose their symbol (X or O).

    Returns:
        tuple: A tuple containing the computer's symbol and the computer's 2 symbol.
    """
    hchoice = ''
    while hchoice != 'O' and hchoice != 'X':
        hchoice = input('Choose X or O\nChosen: ').upper()
    return ('O', 'X') if hchoice == 'X' else ('X', 'O')


def choose_first():
    """
    Decide whether the computer 2 or the computer makes the first move.

    Returns:
        str: 'Y' if computer 2 goes first, 'N' if computer goes first.
    """
    first = ''
    while first != 'Y' and first != 'N':
        first = input('First to start?[y/n]: ').upper()
    return first


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
    """
    if gameover(board, PC):  # Check if IA MINMAX 1 wins
        print('PLAYER IA MINMAX 1 WINS!')
    elif gameover(board, PC2):  # Check if IA MINMAX 2 wins
        print('PLAYER IA MINMAX 2 WINS!')
    else:
        print('DRAW!')

    exit()


def iaturn1(p1_choice, p2_choice):
    """
    Executes the AI player's turn for Player 1 (Minimax) in the game.

    Parameters:
        p1_choice (str): The symbol for the first player ('X' or 'O').
        p2_choice (str): The symbol for the second player ('X' or 'O').
    """
    depth = len(clearcells(board))
    if depth == 0 or gameovergame(board):
        return

    player_symbol = p1_choice
    print(f'Player IA MINMAX 1 turn [{player_symbol}]')
    render(board, p1_choice, p2_choice)

    move = minimax(board, depth, PC)  # PC is the player index for Player 1
    x, y = move[0], move[1]

    domove(x, y, PC)
    time.sleep(1)


def iaturn2(p1_choice, p2_choice):
    """
    Executes the AI player's turn for Player 2 (Minimax) in the game.

    Parameters:
        p1_choice (str): The symbol for the first player ('X' or 'O').
        p2_choice (str): The symbol for the second player ('X' or 'O').
    """
    depth = len(clearcells(board))
    if depth == 0 or gameovergame(board):
        return

    player_symbol = p2_choice
    print(f'Player IA MINMAX 2 turn [{player_symbol}]')
    render(board, p1_choice, p2_choice)

    move = minimax(board, depth, PC2)  # PC2 is the player index for Player 2
    x, y = move[0], move[1]

    domove(x, y, PC2)
    time.sleep(1)


if __name__ == '__main__':
    main()
