# Min max algorithm for TIC TAC TOE!!!

from math import inf as infinity

board = [
    [ 0, 0, 0],
    [ 0, 0, 0],
    [ 0, 0, 0],
]

PC = +1
HUMAN = -1


def evaluate(state):
    """
    """
    if (gameover(state, PC)):
        score = +1
    elif (gameover(state, HUMAN)):
        score = -1
    else:
        score = 0
    return score

def eval_line(r1, c1, r2, c2, r3, c3):
    """
    """
    score = 0
    '''
    First cell case
    Lets supose X is PC and O is a Human
    So, we have:
    X ? ?; score = +1
    O ? ?; score = -1
    ? ? ?; score = 0
    '''
    if (PC == board[r1][c1]):
        score = +1
    elif (HUMAN == board[r1][c1]):
        score == -1

    '''
    Second cell case
    Lets supose X is PC and O is a Human
    So, we have:
    X X ?; score = +10 because it has more chances to win
    O O ?; score = -10 because it has more chances to win
    ? ? ?; score = 0
    '''
    if (PC == board[r2][c2]):
        if (score == 1):
            score = +10     # X X ?; it has chances to win
        elif (score == -1):
            return 0        # X O ?; it has no chances to win
        else:
            score = +1      # ? X ?; cell 1 is empty
    elif (HUMAN == board[r2][c2]):
        if (score == -1):
            score = -10     # O O ?; it has chances to win
        elif (score == 1):
            return 0        # X O ?; draw
        else:
            score = -1      # ? O ?; cell 1 is empty

    '''
    Third cell case
    Lets supose X is PC and O is a Human
    So, we have:
    X X X; score = +100 win
    O O O; score = -100 lose
    ? ? ?; score = 0
    '''
    if (PC == board[r3][c3]):
        if (score > 0):
            score *=10      # X X X (+100) or ? X X (+10)
        elif (score < 0):
            return 0        # X X O or O ? X; draw
        else:
            score = +1      # ? ? X; cell 2 was empty
    elif (HUMAN == board[r3][c3]):
        if (score < 0):
            score *=10      # O O O (-100) or ? O O (-10)
        elif (score > 1):
            return 0        # X X O or X ? O; draw
        else:
            score = -1      # ? ? O; cell 2 was empty

    return score

def gameover(state, player):
    """
    """
    winstate = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        [state[2][0], state[1][1], state[0][2]]
    ]
    if([player, player, player] in winstate):
        return True
    else:
        return False
    
def clear_cells(state):
    """
    """
    cells = []
    for i, row in enumerate(state):
        for j, cell in enumerate(row):
            if (cell == 0):
                cells.append([i, j])
    return cells
    
def minimax(state, depth, player):
    if depth == 0 or gameover(state, player):
        score = evaluate(state)
        return [-1, -1, score]

# Minimal test
# print('Results', evaluate())
print('Score: ', gameover(board, 1))