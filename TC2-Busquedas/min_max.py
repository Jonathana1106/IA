# Min max algorithm for TIC TAC TOE!!!
import platform
import time
import os
from math import inf as infinity
from random import choice


######################################################### Game logic ##############################################################
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
        [state[2][0], state[1][1], state[0][2]]
    ]
    if ([player, player, player] in winstate):
        return True
    else:
        return False


def domove(i, j, player):
    """
    Make a move on the game board for the human player if it's a valid move.

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
        # If the current player is the human, minimize the score.
        else:
            if score[2] < best[2]:
                best = score

    return best


######################################################### Game interface ##############################################################

def main():
    """
    Main function to run the Tic Tac Toe game.
    """
    clean()
    hchoice = ''  # X or O
    pchoice = ''  # X or O
    first = ''  # if human is the first

    # Human chooses X or O to play
    while hchoice != 'O' and hchoice != 'X':
        try:
            print('')
            hchoice = input('Choose X or O\nChosen: ').upper()
        except (EOFError, KeyboardInterrupt):
            print('Bye')
            exit()
        except (KeyError, ValueError):
            print('Bad choice')

    # Setting computer's choice
    if hchoice == 'X':
        pchoice = 'O'
    else:
        pchoice = 'X'

    # Human may start first
    clean()
    while first != 'Y' and first != 'N':
        try:
            first = input('First to start?[y/n]: ').upper()
        except (EOFError, KeyboardInterrupt):
            print('Bye')
            exit()
        except (KeyError, ValueError):
            print('Bad choice')

    # Main loop of this game
    while len(clearcells(board)) > 0 and not gameovergame(board):
        if first == 'N':
            iaturn(pchoice, hchoice)
            first = ''

        humanturn(pchoice, hchoice)
        iaturn(pchoice, hchoice)

    # Game over message
    if gameover(board, HUMAN):
        clean()
        print(f'Human turn [{hchoice}]')
        render(board, pchoice, hchoice)
        print('YOU WIN!')
    elif gameover(board, PC):
        clean()
        print(f'Computer turn [{pchoice}]')
        render(board, pchoice, hchoice)
        print('YOU LOSE!')
    else:
        clean()
        render(board, pchoice, hchoice)
        print('DRAW!')

    exit()


def clean():
    """
    Clears the terminal/console screen.
    """
    osname = platform.system().lower()
    if 'windows' in osname:
        os.system('cls')
    else:
        os.system('clear')


def render(state, pchoice, hchoice):
    """
    Renders the current state of the game board on the console.

    Parameters:
        state (list[list[int]]): The current game board state.
        pchoice (str): The computer's choice ('X' or 'O').
        hchoice (str): The human's choice ('X' or 'O').
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


def iaturn(pchoice, hchoice):
    """
    Executes the computer's turn in the game.

    Parameters:
        pchoice (str): The computer's choice ('X' or 'O').
        hchoice (str): The human's choice ('X' or 'O').
    """
    depth = len(clearcells(board))
    if depth == 0 or gameovergame(board):
        return

    clean()
    print(f'Computer turn [{pchoice}]')
    render(board, pchoice, hchoice)

    if depth == 9:
        x = choice([0, 1, 2])
        y = choice([0, 1, 2])
    else:
        move = minimax(board, depth, PC)
        x, y = move[0], move[1]

    domove(x, y, PC)
    time.sleep(1)


def humanturn(pchoice, hchoice):
    """
    Executes the human's turn in the game.

    Parameters:
        pchoice (str): The computer's choice ('X' or 'O').
        hchoice (str): The human's choice ('X' or 'O').
    """
    depth = len(clearcells(board))
    if depth == 0 or gameovergame(board):
        return

    # Dictionary of valid moves
    move = -1
    moves = {
        1: [0, 0], 2: [0, 1], 3: [0, 2],
        4: [1, 0], 5: [1, 1], 6: [1, 2],
        7: [2, 0], 8: [2, 1], 9: [2, 2],
    }

    clean()
    print(f'Human turn [{hchoice}]')
    render(board, pchoice, hchoice)

    while move < 1 or move > 9:
        try:
            move = int(input('Use numpad (1..9): '))
            coord = moves[move]
            canmove = domove(coord[0], coord[1], HUMAN)

            if not canmove:
                print('Bad move')
                move = -1
        except (EOFError, KeyboardInterrupt):
            print('Bye')
            exit()
        except (KeyError, ValueError):
            print('Bad choice')


if __name__ == '__main__':
    main()
