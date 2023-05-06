# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part A: Single Player Infexion
from queue import PriorityQueue
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


# all function below are for a DUMB players (Blue and RED)
# essentially random moves regardless of player
# need to turn them to greedy moves with use of heuristic

def spawn(board: dict[tuple, tuple], coord: tuple, player: str, enemy):
    if coord in board:
        while coord in board:
            coord = (random.randint(0, 6), random.randint(0, 6))
    board[coord] = (player, 1)
    return SpawnAction(HexPos(coord[0], coord[1]))

def make_move(input: dict[tuple, tuple], player: str, enemy):
    playerCell_list = coord_list(input, player)
    enemyCell_list = coord_list(input, enemy)
    total_power = count_power(input)
    distance_dict = {}

    for playerCell in playerCell_list:
        enemyCell = find_closest_cell(input, playerCell, enemy)
        if enemyCell is not None:
            distance = travel_distance(*playerCell, *enemyCell, input[playerCell][1])
            distance_dict[playerCell] = (enemyCell, distance)

    if distance_dict:
        playerCell = min(distance_dict, key=lambda k: distance_dict[k][1])
        direction = determine_direction(playerCell, distance_dict[playerCell][0])

        if len(playerCell_list) == 1:
            if total_power >= 48:
                return simple_spread(input, playerCell, direction)
            return spawn(input, (random.randint(0, 6), random.randint(0, 6)), player, enemy)
        return simple_spread(input, playerCell, direction)


def simple_spread(board: dict[tuple, tuple], playerCell, direction):
    return SpreadAction(HexPos(playerCell[0], playerCell[1]), HexDir(direction))

def coord_list(input, player):
    result_list = list()

    for cell in input.keys():
        if input[cell][0] == player:
            result_list.append(cell)
    return result_list

def find_closest_cell(input: dict[tuple,tuple], cell: tuple, enemy):
    directions = [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)]
    visited = []
    queue = PriorityQueue()
    # add start cell with distance 0
    queue.put((0, cell))

    # A* (kind of)search algorithm
    while queue:
        current_dist, current_cell = queue.get()
        if current_cell in input:
            if input[current_cell][0] == enemy:
                return current_cell

        for d in directions:
            next_cell = determine_next_cell(current_cell, d)
            if next_cell not in visited:
                visited.append(next_cell)
                # calculate the distance to the target cell using a heuristic function
                # estimated_dist = calculate_heuristic(next_cell, input, enemy)
                estimated_dist = current_dist + 1 + calculate_heuristic(next_cell, input, enemy)
                # add the cell to the priority queue using heuristic as priority
                queue.put((estimated_dist, next_cell))


def determine_next_cell(current_cell: tuple, direction: tuple):
    # hex directions: (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)

    current_x = current_cell[0]
    current_y = current_cell[1]

    next_x = current_x + direction[0]
    next_y = current_y + direction[1]

    # check if the coordinates wrap around the board
    if next_x > 6:
        next_x -= 7
    elif next_x < 0:
        next_x += 7
    if  next_y > 6:
        next_y -= 7
    elif next_y < 0:
        next_y += 7

    return (next_x, next_y)



def calculate_heuristic(cell: tuple, input: dict[tuple, tuple], enemy):
    enemyCell_list = coord_list(input, enemy)

    # print("blueCell_list")
    # print(blueCell_list)

    distance_dict = dict()
    for enemy_cell in enemyCell_list:
        distance = calc_distance(cell[0], cell[1], enemy_cell[0], enemy_cell[1])

        # print("distance")
        # print(distance)

        distance_dict[enemy_cell] = distance
    min_distance_cell = min(distance_dict, key=distance_dict.get)
    return distance_dict[min_distance_cell]

def calc_distance(x1, y1, x2, y2):
    # distance of two points
    # calculate new distance if wrap around
    dx, dy = wrap_check((x1 - x2), (y1 - y2))

    # hexoganol distance check if points in a straight line (fastest route)
    if (x1 < x2 and y1 < y2) or (x1 > x2 and y1 > y2):
        return abs(dx) + abs(dy)
    else:
        return max(abs(dx), abs(dy))

def wrap_check(dx, dy):
    dx_wrap = 100
    dy_wrap = 100

    # check for wrap around
    if dx > 3.5:
        dx_wrap = 7 - dx
    elif dx < -3.5:
        dx_wrap = 7 + dx

    if dy > 3.5:
        dy_wrap = 7 - dy
    elif dy < -3.5:
        dy_wrap = 7 + dy

    # return the minimum of the two
    return min(dx, dx_wrap), min(dy, dy_wrap)

def travel_distance(x1, y1, x2, y2, power):
    # how long I should travel in a certain direction until x1=x2 and y1=y2
    direction = determine_direction((x1, y1), (x2, y2))
    reached = False
    cell = (x1, y1)
    distance = 0
    next_cell = determine_next_cell(cell, (direction[0] * power, direction[1] * power))

    while not reached:
        distance += 1
        if next_cell == (x2, y2):
            reached = True
            break
        direction = determine_direction(next_cell, (x2, y2))
        next_cell = determine_next_cell(next_cell, direction)
    return distance

def determine_direction(start: tuple, target: tuple):
    x1, x2 = start[0], target[0]
    y1, y2 = start[1], target[1]

    dx, dy = wrap_check(x2 - x1, y2 - y1)

    # determine direction to take
    if dx == 0:
        if dy < 0:
            return (0, -1)  # north-west
        else:
            return (0, 1)  # south-east
    elif dx > 0:
        if dy >= 0:
            if dx > dy:
                return (1, 0)  # north-east
            else:
                return (0, 1)  # south-east
        else:
            return (1, -1)  # north
    else:  # dx < 0
        if dy <= 0:
            if dx > dy:
                return (0, -1)  # north-west
            else:
                return (-1, 0)  # south-west
        else:  # dy > 0
            return (-1, 1)  # south

def spread(input: dict[tuple, tuple], action: tuple, colour):
    cell = (action[0], action[1])
    direction = (action[2], action[3])
    if cell in input:
        power = input[cell][1]
        current_cell = cell

        # iterate through the cells that are to be covered, and check for blue cells
        for i in range(power):
            next_cell = determine_next_cell(current_cell, direction)
            if next_cell in input:
                next_power = input[next_cell][1]
                # if cell to spread to has power of 6, empty cell
                if next_power == 6:
                    input.pop(next_cell)
                # if blue cell -> capture, if red cell -> update cells to reflect board state
                else:
                    input[next_cell] = (colour, next_power + 1)
            else:
                input[next_cell] = (colour, 1)
            current_cell = next_cell
        # update cell -> empty it
        input.pop(cell)

def count_power(board: dict[tuple, tuple]):
    total_power = 0
    values = list(board.values())
    for index, tuple in enumerate(values):
        total_power += values[index][1]
    return total_power

