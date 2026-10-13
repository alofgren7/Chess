"""Piece hierarchy and per-piece move generation.

Design
------
Each piece type knows how *it* moves, and nothing else. The board knows what is
where, and the game knows the rules that sit above the board. That split is what
keeps this module free of any check/checkmate logic: a piece never asks "would
this expose my king?", it only answers "where could I go if the king didn't
exist?". The board filters the illegal ones out afterwards.

The important subtlety is the split between two questions:

``attacked_squares``
    Which squares does this piece bear down on? Used for check detection and
    for castling's "the king may not pass through an attacked square" rule.
    This intentionally *includes* squares occupied by friendly pieces (those
    squares are defended, and it costs nothing to report them, since the only
    square ever queried for check is the enemy king's).

``generate_moves``
    Which moves could this piece actually play? Usually just
    ``attacked_squares`` minus friendly-occupied squares.

For every piece except the pawn these two are nearly identical, which is why the
base class can implement ``generate_moves`` once in terms of ``attacked_squares``.
The pawn is the exception that justifies the whole split: it *moves* straight
forward but *attacks* diagonally, so conflating the two would let pawns give
check down a file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, TYPE_CHECKING

from .move import Move, Square, in_bounds

if TYPE_CHECKING:  # pragma: no cover
    # Imported for type checking only: board.py imports this module, so a real
    # import here would be circular at runtime.
    from .board import Board

WHITE = "white"
BLACK = "black"


def opponent(color: str) -> str:
    return BLACK if color == WHITE else WHITE


class Piece(ABC):
    """Base class for every chess piece.

    A piece stores only what is intrinsic to it: its colour and whether it has
    ever moved. It deliberately does *not* store its own square. Position is the
    board's responsibility, so a piece is passed its square whenever it needs
    one. Keeping position out of the piece means ``make_move``/``undo_move`` can
    shuffle pieces around the grid without keeping a second copy of the truth in
    sync, which is a classic source of bugs.

    ``has_moved`` exists purely to serve castling rights (both for the king and
    for each rook) and pawn double-pushes. It is saved and restored by the undo
    machinery, because a move that is tried and taken back must not silently
    destroy castling rights.
    """

    letter = "?"  # Uppercase piece letter used by SAN and FEN
    points = 0  # Conventional material value, used for the GUI's advantage readout
    _glyphs = ("?", "?")  # (white, black) unicode glyphs

    def __init__(self, color: str) -> None:
        self.color = color
        self.has_moved = False

    @property
    def is_white(self) -> bool:
        return self.color == WHITE

    @property
    def symbol(self) -> str:
        """Uppercase for white, lowercase for black."""
        return self.letter.upper() if self.is_white else self.letter.lower()

    @property
    def glyph(self) -> str:
        return self._glyphs[0] if self.is_white else self._glyphs[1]

    @abstractmethod
    def attacked_squares(self, board: "Board", square: Square) -> Iterator[Square]:
        """Squares this piece bears down on, ignoring king safety.

        Returns an iterator rather than a list so callers that only need to know
        "does this attack square X?" can bail out early without building the
        whole set. ``Board.is_attacked`` runs for every generated move, so this
        adds up.
        """

    def generate_moves(self, board: "Board", square: Square) -> Iterator[Move]:
        """Pseudo-legal moves: correct piece movement, king safety not checked.

        "Pseudo-legal" is the standard term for a move that respects how the
        piece moves but might leave its own king in check. Generating these
        first and filtering afterwards is far simpler than trying to compute
        pins and discovered checks up front, and it is impossible to get subtly
        wrong -- see ``Board.legal_moves``.
        """
        for target in self.attacked_squares(board, square):
            occupant = board.piece_at(target)
            # An empty square or an enemy piece; friendly pieces block.
            if occupant is None or occupant.color != self.color:
                yield Move(square, target)

    def _ray_targets(
        self, board: "Board", square: Square, directions
    ) -> Iterator[Square]:
        """Walk outwards along each direction until something blocks the ray.

        Shared by bishop, rook and queen -- the only difference between those
        three pieces is the set of directions, so the traversal is written once.

        The blocking square is yielded *before* breaking: an enemy piece there
        can be captured, and a friendly piece there is defended (which matters
        for attack maps). ``generate_moves`` is what discards the friendly case.
        """
        row, col = square
        for d_row, d_col in directions:
            r, c = row + d_row, col + d_col
            while in_bounds(r, c):
                yield (r, c)
                if board.piece_at((r, c)) is not None:
                    break
                r += d_row
                c += d_col

    def _step_targets(self, square: Square, offsets) -> Iterator[Square]:
        """Single-hop moves; shared by knight and king.

        No board lookup is needed because these pieces cannot be blocked on the
        way to a square -- only on the square itself, which the caller handles.
        """
        row, col = square
        for d_row, d_col in offsets:
            r, c = row + d_row, col + d_col
            if in_bounds(r, c):
                yield (r, c)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.color})"


class Pawn(Piece):
    """The awkward one: it is the only piece whose moves differ from its attacks,
    the only one that cannot move backwards, and the only one involved in
    promotion, double pushes and en passant."""

    letter = "P"
    points = 1
    _glyphs = ("\u2659", "\u265f")

    @property
    def direction(self) -> int:
        """+1 for White, -1 for Black (see the coordinate note in move.py)."""
        return 1 if self.is_white else -1

    @property
    def start_row(self) -> int:
        return 1 if self.is_white else 6

    @property
    def promotion_row(self) -> int:
        return 7 if self.is_white else 0

    def attacked_squares(self, board: "Board", square: Square) -> Iterator[Square]:
        # Diagonally forward only, and reported whether or not anything is there:
        # an empty diagonal square is still *controlled* by this pawn, which is
        # what check detection and castling safety checks care about.
        row, col = square
        for d_col in (-1, 1):
            r, c = row + self.direction, col + d_col
            if in_bounds(r, c):
                yield (r, c)

    def generate_moves(self, board: "Board", square: Square) -> Iterator[Move]:
        # Overridden wholesale rather than extending the base implementation,
        # because for a pawn almost nothing about the default logic applies.
        row, col = square

        # Forward pushes: strictly onto empty squares, never captures.
        step = (row + self.direction, col)
        if in_bounds(*step) and board.piece_at(step) is None:
            yield from self._promotions(square, step)
            # The double push is only legal if the single push was, which is why
            # it is nested here -- it cannot jump over an occupied square.
            double = (row + 2 * self.direction, col)
            if row == self.start_row and board.piece_at(double) is None:
                yield Move(square, double, is_double_push=True)

        # Captures: the diagonals, which are only moves when something is there.
        for target in self.attacked_squares(board, square):
            victim = board.piece_at(target)
            if victim is not None and victim.color != self.color:
                yield from self._promotions(square, target)
            elif victim is None and target == board.en_passant_target:
                # En passant: the destination square is empty, and the pawn being
                # captured sits beside us rather than on the destination. The
                # board sets en_passant_target for exactly one ply after a double
                # push, so simply comparing against it enforces the timing rule.
                yield Move(square, target, is_en_passant=True)

    def _promotions(self, start: Square, end: Square) -> Iterator[Move]:
        """Expand a pawn move into four moves when it reaches the last rank.

        Promotion choice is part of the move, not a follow-up action. Modelling
        it that way means the legality filter, SAN generation and undo all work
        on promotions with no special casing -- and the UI's "which piece?"
        prompt is just a choice between four already-legal moves.
        """
        if end[0] == self.promotion_row:
            for piece_letter in ("q", "r", "b", "n"):
                yield Move(start, end, promotion=piece_letter)
        else:
            yield Move(start, end)


class Knight(Piece):
    letter = "N"
    points = 3
    _glyphs = ("\u2658", "\u265e")
    # All eight L-shapes; unlike sliders these are single hops that ignore
    # anything in between, so there is no ray to walk.
    OFFSETS = ((2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1))

    def attacked_squares(self, board: "Board", square: Square) -> Iterator[Square]:
        return self._step_targets(square, self.OFFSETS)


class Bishop(Piece):
    letter = "B"
    points = 3
    _glyphs = ("\u2657", "\u265d")
    DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1))

    def attacked_squares(self, board: "Board", square: Square) -> Iterator[Square]:
        return self._ray_targets(board, square, self.DIRECTIONS)


class Rook(Piece):
    letter = "R"
    points = 5
    _glyphs = ("\u2656", "\u265c")
    DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def attacked_squares(self, board: "Board", square: Square) -> Iterator[Square]:
        return self._ray_targets(board, square, self.DIRECTIONS)


class Queen(Piece):
    letter = "Q"
    points = 9
    _glyphs = ("\u2655", "\u265b")
    # A queen is literally a rook plus a bishop, so reuse rather than restate.
    DIRECTIONS = Rook.DIRECTIONS + Bishop.DIRECTIONS

    def attacked_squares(self, board: "Board", square: Square) -> Iterator[Square]:
        return self._ray_targets(board, square, self.DIRECTIONS)


class King(Piece):
    letter = "K"
    points = 0  # Never captured, so it has no material value to count
    _glyphs = ("\u2654", "\u265a")
    OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

    def attacked_squares(self, board: "Board", square: Square) -> Iterator[Square]:
        # Castling is deliberately excluded here. attacked_squares feeds check
        # detection, and including castling would make is_attacked recurse into
        # the castling rules, which themselves call is_attacked -- infinite loop.
        return self._step_targets(square, self.OFFSETS)

    def generate_moves(self, board: "Board", square: Square) -> Iterator[Move]:
        yield from super().generate_moves(board, square)
        yield from self._castling_moves(board, square)

    def _castling_moves(self, board: "Board", square: Square) -> Iterator[Move]:
        """Castling lives on the king because the king is what the player moves.

        All five FIDE conditions are checked here:

        1. Neither the king nor that rook has ever moved (``has_moved``).
        2. The squares between them are empty.
        3. The king is not currently in check.
        4. The king does not pass through an attacked square.
        5. The king does not land on an attacked square.

        Conditions 3-5 are tested explicitly rather than left to the usual
        legality filter, because that filter only verifies the *final* position.
        It would happily allow a king to castle out of, or straight through,
        check. This is the one rule where the generic "make it and see" approach
        is not sufficient on its own.
        """
        home_row = 0 if self.is_white else 7
        # The square check also guards against a king that was placed on e1/e8
        # by a FEN import but is not actually on its home square.
        if self.has_moved or square != (home_row, 4):
            return
        enemy = opponent(self.color)
        if board.is_attacked(square, enemy):
            return
        # (rook file, direction the king travels, files that must be empty)
        # Queenside needs b1/b8 empty too: the rook passes over it, even though
        # the king never does and so it need not be safe from attack.
        for rook_col, step, empty_cols in ((7, 1, (5, 6)), (0, -1, (1, 2, 3))):
            rook = board.piece_at((home_row, rook_col))
            if not isinstance(rook, Rook) or rook.color != self.color or rook.has_moved:
                continue
            if any(board.piece_at((home_row, c)) is not None for c in empty_cols):
                continue
            crossing = (home_row, 4 + step)
            destination = (home_row, 4 + 2 * step)
            if board.is_attacked(crossing, enemy) or board.is_attacked(destination, enemy):
                continue
            # The rook's destination is always the square the king crossed.
            yield Move(
                square,
                destination,
                is_castle=True,
                rook_start=(home_row, rook_col),
                rook_end=crossing,
            )


# Maps a promotion letter to the class to create. Defined after the classes so
# it can reference them, and shared by the board (when playing a promotion) and
# the GUI (when drawing the four choices).
PROMOTION_CLASSES = {"q": Queen, "r": Rook, "b": Bishop, "n": Knight}
