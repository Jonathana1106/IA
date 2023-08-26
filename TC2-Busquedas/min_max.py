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

def gameovergame(state):
    """
    """
    return gameover(state, HUMAN) or gameover(state, PC)

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

def domove(i, j):
    """
    """
    if validmove(i, j):
        board[i][j] = HUMAN

def validmove(i, j):
    """
    """
    if ([i, j] in clear_cells(board)):
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
    """
    """
    if (player == PC):
        best = [-1, -1, -infinity]
    else:
        best = [-1, -1, +infinity]
    
    if (depth == 0 or gameovergame(state)):
        score = evaluate(state)
        return [-1, -1, score]
    
    for cell in clear_cells(state):
        i, j = cell[0], cell[1]
        state[i][j] = player
        score = minimax(state, depth - 1, -player)
        state[i][j] = 0
        score[0], score[1] = i, j

        if (player == PC):
            if (score[2] > best[2]):
                best = score
        else:
            if (score[2] < best[2]):
                best = score
    return best