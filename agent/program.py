# COMP30024 Artificial Intelligence, Semester 1 2023
# Project Part B: Game Playing Agent

from referee.game import \
    PlayerColor, Action, SpawnAction, SpreadAction, HexPos, HexDir
from .utils import spawn, make_move, spread


# This is the entry point for your game playing agent. Currently the agent
# simply spawns a token at the centre of the board if playing as RED, and
# spreads a token at the centre of the board if playing as BLUE. This is
# intended to serve as an example of how to use the referee API -- obviously
# this is not a valid strategy for actually playing the game!

import random

class Agent:

    board = {}

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initialise the agent.
        """

        self._color = color
        self._round = 0
        

        # determine Player
        match color:
            case PlayerColor.RED:
                self._player = "RED"
                self._enemy = "BLUE"
                print("Testing: I am playing as red")
            case PlayerColor.BLUE:
                self._player = "BLUE"
                self._enemy = "RED"
                print("Testing: I am playing as blue")

    def action(self, **referee: dict) -> Action:
        """
        Return the next action to take.
        """

        

        if (self._round == 0 or self._round == 1): #first move
            self._round += 1
            return spawn(self.board, (random.randint(0, 6), random.randint(0, 6)), self._player, self._enemy)
        else:
            self._round += 1
            return make_move(self.board, self._player, self._enemy)

        # match self._color:
        #     case PlayerColor.RED:
        #         return SpawnAction(HexPos(3, 3))
        #     case PlayerColor.BLUE:
        #         # This is going to be invalid... BLUE never spawned!
        #         return SpreadAction(HexPos(3, 3), HexDir.Up)

    def turn(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Update the agent with the last player's action.
        """

        match action:
            case SpawnAction(cell):

                self.board[cell.r, cell.q] = (color.name, 1)

                pass
            case SpreadAction(cell, direction):

                direction_r = direction.value.r
                direction_q = direction.value.q

                spread(self.board, (cell.r, cell.q, direction_r, direction_q), color.name)

                print(f"Testing: {color} SPREAD from {cell}, {direction}")
                pass


