# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part A: Single Player Infexion

import random

from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir

def apply_ansi(str, bold=True, color=None):
    """
    Wraps a string with ANSI control codes to enable basic terminal-based
    formatting on that string. Note: Not all terminals will be compatible!

    Arguments:

    str -- String to apply ANSI control codes to
    bold -- True if you want the text to be rendered bold
    color -- Colour of the text. Currently only red/"r" and blue/"b" are
        supported, but this can easily be extended if desired...

    """
    bold_code = "\033[1m" if bold else ""
    color_code = ""
    if color == "r":
        color_code = "\033[31m"
    if color == "b":
        color_code = "\033[34m"
    return f"{bold_code}{color_code}{str}\033[0m"


def render_board(board: dict[tuple, tuple], ansi=False) -> str:
    """
    Visualise the Infexion hex board via a multiline ASCII string.
    The layout corresponds to the axial coordinate system as described in the
    game specification document.

    Example:

        >>> board = {
        ...     (5, 6): ("r", 2),
        ...     (1, 0): ("b", 2),
        ...     (1, 1): ("b", 1),
        ...     (3, 2): ("b", 1),
        ...     (1, 3): ("b", 3),
        ... }
        >>> print_board(board, ansi=False)

                                ..
                            ..      ..
                        ..      ..      ..
                    ..      ..      ..      ..
                ..      ..      ..      ..      ..
            b2      ..      b1      ..      ..      ..
        ..      b1      ..      ..      ..      ..      ..
            ..      ..      ..      ..      ..      r2
                ..      b3      ..      ..      ..
                    ..      ..      ..      ..
                        ..      ..      ..
                            ..      ..
                                ..
    """
    dim = 7
    output = ""
    for row in range(dim * 2 - 1):
        output += "    " * abs((dim - 1) - row)
        for col in range(dim - abs(row - (dim - 1))):
            # Map row, col to r, q
            r = max((dim - 1) - row, 0) + col
            q = max(row - (dim - 1), 0) + col
            if (r, q) in board:
                color, power = board[(r, q)]
                text = f"{color}{power}".center(4)
                if ansi:
                    output += apply_ansi(text, color=color, bold=False)
                else:
                    output += text
            else:
                output += " .. "
            output += "    "
        output += "\n"
    return output


def coord_list(input, player):
    result_list = list()
    for cell in input.keys():
        if input[cell][0] == player:
            result_list.append(cell)
    return result_list


# all function below are for a DUMB players (Blue and RED)
# essentially random moves regardless of player
# need to turn them to greedy moves with use of heuristic

def spawn(board: dict[tuple, tuple], coord: tuple, player: str):

    if coord in board:
        coord = (random.randint(0, 6), random.randint(0, 6))
    board[coord] = (player, 1)
    return SpawnAction(HexPos(coord[0], coord[1]))

def make_move(input: dict[tuple, tuple], player: str):
    cell_list = coord_list(input, player)
    if len(cell_list) == 1:
        input[(cell_list[0][0]+1, cell_list[0][1]+1)] = (player, 1)
        return SpawnAction(HexPos(cell_list[0][0]+1, cell_list[0][1]+1)) # will eventually go out of bounds
    else:
        return simple_spread(input, player)


def simple_spread(board: dict[tuple, tuple], color: 'PlayerColor'):
    print(list(board.keys()))
    blind_pick = list(board.keys())[random.randint(0, len(list(board.keys())))]
    blind_direction = list(HexDir)[random.randint(0, len(list(HexDir)))]
    return SpreadAction(HexPos(blind_pick[0], blind_pick[1]), blind_direction)


