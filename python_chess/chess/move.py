"""Square helpers and the :class:`Move` value object.

Coordinate system
-----------------
A square is a plain ``(row, col)`` tuple of integers 0-7.

* ``row`` 0 is rank 1 (White's back rank) and row 7 is rank 8.
* ``col`` 0 is file a and col 7 is file h.

That orientation is deliberate: it makes White's pawns move in the ``+row``
direction, so "forward" is ``+1`` for White and ``-1`` for Black. Many chess
programs put rank 8 at row 0 to match FEN's reading order, but then every pawn
calculation is inverted relative to how players think. Flipping the board for
display is a rendering concern and is handled in the UI layer instead.

Tuples (rather than a Square class) keep squares hashable, cheap to create and
trivially comparable, which matters because move generation creates millions of
them during a perft run.
"""

from __future__ import annotations

from typing import Optional, Tuple

# A square is (row, col). Aliased so signatures read as intent, not as geometry.
Square = Tuple[int, int]

FILES = "abcdefgh"
RANKS = "12345678"


def in_bounds(row: int, col: int) -> bool:
    """True when the coordinates land on the board.

    Takes two ints rather than a tuple because move generators call it in tight
    loops on freshly computed values, and unpacking a tuple there would be pure
    overhead.
    """
    return 0 <= row < 8 and 0 <= col < 8


def square_name(square: Square) -> str:
    """``(3, 4)`` -> ``'e4'``."""
    row, col = square
    return f"{FILES[col]}{RANKS[row]}"


def parse_square(text: str) -> Square:
    """``'e4'`` -> ``(3, 4)``; raises ValueError on anything else.

    This is the single place where user text becomes board coordinates, so all
    validation lives here and callers downstream can trust what they receive.
    """
    text = text.strip().lower()
    if len(text) != 2 or text[0] not in FILES or text[1] not in RANKS:
        raise ValueError(f"'{text}' is not a valid square")
    return (RANKS.index(text[1]), FILES.index(text[0]))


class Move:
    """A single candidate move, independent of whether it is legal.

    A Move is a *description* of an action, not a record of one. It carries no
    reference to the board or to the piece being moved, which means the same
    object can be generated, tested for legality, discarded, and regenerated
    without any stale state. Everything needed to undo a move is captured
    separately by the board at the moment it is played (see ``Board._UndoState``).

    The special-case flags exist because each one requires the board to touch a
    square other than ``start``/``end``:

    * ``is_castle`` also moves a rook, so it carries ``rook_start``/``rook_end``.
    * ``is_en_passant`` captures a pawn that is *not* on ``end``.
    * ``is_double_push`` is what creates an en passant opportunity next ply.
    * ``promotion`` replaces the pawn with a new piece on arrival.

    Storing these as flags decided at generation time means ``make_move`` never
    has to re-derive intent from geometry (e.g. guessing "a king moved two files,
    so this must be castling"), which is both faster and less error-prone.
    """

    # Moves are created in huge numbers during search/perft, so __slots__ drops
    # the per-instance dict and measurably reduces memory churn.
    __slots__ = (
        "start",
        "end",
        "promotion",
        "is_castle",
        "is_en_passant",
        "is_double_push",
        "rook_start",
        "rook_end",
    )

    def __init__(
        self,
        start: Square,
        end: Square,
        promotion: Optional[str] = None,
        is_castle: bool = False,
        is_en_passant: bool = False,
        is_double_push: bool = False,
        rook_start: Optional[Square] = None,
        rook_end: Optional[Square] = None,
    ) -> None:
        self.start = start
        self.end = end
        self.promotion = promotion  # 'q', 'r', 'b' or 'n', else None
        self.is_castle = is_castle
        self.is_en_passant = is_en_passant
        self.is_double_push = is_double_push
        self.rook_start = rook_start
        self.rook_end = rook_end

    @property
    def uci(self) -> str:
        """Long algebraic form, e.g. ``'e2e4'`` or ``'e7e8q'``.

        This is the format the console UI accepts and the GUI shows in move
        lists. It is unambiguous without board context, unlike SAN.
        """
        return square_name(self.start) + square_name(self.end) + (self.promotion or "")

    def __eq__(self, other: object) -> bool:
        # Equality deliberately ignores the special-case flags: start, end and
        # promotion uniquely identify a move within a given position, and the
        # flags are derived facts about that same move. This lets a move parsed
        # from user input match the fully-populated one from the generator.
        if not isinstance(other, Move):
            return NotImplemented
        return (
            self.start == other.start
            and self.end == other.end
            and self.promotion == other.promotion
        )

    def __hash__(self) -> int:
        # Must mirror __eq__ so Moves behave correctly in sets and dict keys.
        return hash((self.start, self.end, self.promotion))

    def __repr__(self) -> str:
        return f"Move({self.uci})"
