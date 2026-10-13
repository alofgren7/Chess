"""A two-player chess implementation with full rule support.

Layers, from the bottom up:

* ``move``   -- squares, coordinates and the Move value object
* ``pieces`` -- how each piece moves
* ``board``  -- position, make/undo, legality, notation
* ``game``   -- players, history, results, and the console UI
* ``gui``    -- the pygame front-end

Each layer depends only on the ones below it in that list. ``gui`` is
deliberately *not* imported here: pygame is optional, and importing this
package must not require it.
"""

from .board import Board
from .game import ConsoleUI, Game, Player
from .move import Move, parse_square, square_name
from .pieces import (
    BLACK,
    WHITE,
    Bishop,
    King,
    Knight,
    Pawn,
    Piece,
    Queen,
    Rook,
)

__all__ = [
    "Board",
    "Game",
    "ConsoleUI",
    "Player",
    "Move",
    "parse_square",
    "square_name",
    "Piece",
    "Pawn",
    "Knight",
    "Bishop",
    "Rook",
    "Queen",
    "King",
    "WHITE",
    "BLACK",
]
