"""Board state, move execution and rule enforcement.

This is the heart of the program. The board owns the grid, the side to move, the
clocks and the en passant square, and it is the only place that mutates any of
them.

The central design decision is the **make / test / unmake** pattern used by
``legal_moves``. Rather than computing pins, discovered attacks and
check-evasions analytically, the board simply plays each pseudo-legal move,
asks whether the mover's king is now attacked, and takes the move back. This is
slower than a purpose-built legal-move generator, but it is dramatically simpler
and -- crucially -- it is correct by construction for every awkward case at once:
pins, discovered checks, king moves along an attacking ray, and the notorious
en-passant-exposes-the-king position. For a two-player game with no search, the
performance cost is irrelevant.

For that pattern to be safe, ``undo_move`` must restore state *perfectly*,
including things that are easy to forget (castling rights, the en passant
square, the halfmove clock). That is what ``_UndoState`` guarantees.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

from .move import Move, Square, parse_square, square_name
from .pieces import (
    BLACK,
    PROMOTION_CLASSES,
    WHITE,
    Bishop,
    King,
    Knight,
    Pawn,
    Piece,
    Queen,
    Rook,
    opponent,
)

# Piece classes in file order (a-file first) for the back rank of both sides.
BACK_RANK = (Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook)


class _UndoState:
    """Everything needed to restore the board after a move.

    A move is *not* reversible from the move object alone: it cannot tell you
    what was captured, what the en passant square used to be, or whether the
    rook it moved had moved before. So the board snapshots those facts onto a
    stack before mutating, and pops them to rewind.

    Note that ``piece`` stores the original object. That matters for promotions:
    the piece placed on the board is a brand new Queen, so undoing has to put
    the *original pawn instance* back rather than construct a fresh one, or the
    identity (and ``has_moved`` history) of that pawn would be lost.

    Snapshotting deltas rather than copying the whole board keeps undo O(1) and
    makes the make/test/unmake loop cheap enough to run for every move.
    """

    __slots__ = (
        "move",
        "piece",
        "captured",
        "captured_square",  # differs from move.end only for en passant
        "piece_had_moved",
        "rook_had_moved",
        "en_passant_target",
        "halfmove_clock",
        "fullmove_number",
        "turn",
    )

    def __init__(self, **kwargs) -> None:
        # Keyword-only construction keeps the call site at make_move readable;
        # with ten positional fields it would be impossible to review.
        for key, value in kwargs.items():
            setattr(self, key, value)


class Board:
    """An 8x8 chess board. ``(row, col)`` with row 0 = rank 1, col 0 = file a.

    Castling rights are not stored as flags on the board. They are derived from
    the ``has_moved`` attribute of the kings and rooks themselves, which means
    there is exactly one copy of that truth and it cannot drift out of sync with
    the pieces. The cost is that FEN import has to reconstruct those flags (see
    ``from_fen``), and that undo must restore them.
    """

    def __init__(self, empty: bool = False) -> None:
        self.squares: List[List[Optional[Piece]]] = [[None] * 8 for _ in range(8)]
        self.turn = WHITE
        # Square a pawn could move *to* when capturing en passant; set for
        # exactly one ply after a double push, then cleared. Storing the target
        # square (rather than the pushed pawn) means the pawn generator can test
        # for it with a single equality check.
        self.en_passant_target: Optional[Square] = None
        self.halfmove_clock = 0  # Plies since the last capture or pawn move
        self.fullmove_number = 1
        self.undo_stack: List[_UndoState] = []
        if not empty:
            self.setup()

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        """Place the standard starting position."""
        for col, piece_class in enumerate(BACK_RANK):
            self.squares[0][col] = piece_class(WHITE)
            self.squares[1][col] = Pawn(WHITE)
            self.squares[6][col] = Pawn(BLACK)
            self.squares[7][col] = piece_class(BLACK)

    @classmethod
    def from_fen(cls, fen: str) -> "Board":
        """Build a position from Forsyth-Edwards Notation.

        Primarily a testing tool: it is what makes it possible to set up the
        standard perft positions and any bug report as a one-line string, rather
        than by playing a sequence of moves to reach it.

        The fiddly part is reconstructing ``has_moved``, since FEN records
        castling *rights* rather than move history. The strategy is to mark
        every imported piece as having moved (the safe default, which forbids
        castling and double pushes), then selectively clear the flag where the
        FEN proves the piece is untouched.
        """
        # Pad so the clock fields are optional; only the first four are needed.
        placement, turn, castling, en_passant = (fen.split() + ["-", "-"])[:4]
        classes = {c.letter: c for c in (Pawn, Knight, Bishop, Rook, Queen, King)}
        board = cls(empty=True)

        # FEN lists ranks from 8 down to 1, so rank_index 0 is our row 7.
        for rank_index, rank in enumerate(placement.split("/")):
            row, col = 7 - rank_index, 0
            for char in rank:
                if char.isdigit():
                    col += int(char)  # a run of empty squares
                    continue
                piece = classes[char.upper()](WHITE if char.isupper() else BLACK)
                piece.has_moved = True
                board.squares[row][col] = piece
                col += 1

        board.turn = WHITE if turn.lower().startswith("w") else BLACK
        board.en_passant_target = (
            None if en_passant == "-" else parse_square(en_passant)
        )

        # A king on its home square is assumed unmoved; the loop below revokes
        # that again if the FEN grants it no castling rights at all.
        for row, color in ((0, WHITE), (7, BLACK)):
            king = board.squares[row][4]
            if isinstance(king, King) and king.color == color:
                king.has_moved = False
        for flag, (row, col) in (("K", (0, 7)), ("Q", (0, 0)), ("k", (7, 7)), ("q", (7, 0))):
            rook = board.squares[row][col]
            if flag in castling and isinstance(rook, Rook):
                rook.has_moved = False
        for row, flags in ((0, "KQ"), (7, "kq")):
            if not any(flag in castling for flag in flags):
                king = board.squares[row][4]
                if isinstance(king, King):
                    king.has_moved = True

        # Pawns still on their starting rank may legally make a double push.
        for row, color in ((1, WHITE), (6, BLACK)):
            for col in range(8):
                pawn = board.squares[row][col]
                if isinstance(pawn, Pawn) and pawn.color == color:
                    pawn.has_moved = False
        return board

    # ------------------------------------------------------------- inspection
    def piece_at(self, square: Square) -> Optional[Piece]:
        row, col = square
        return self.squares[row][col]

    def set_piece(self, square: Square, piece: Optional[Piece]) -> None:
        row, col = square
        self.squares[row][col] = piece

    def all_pieces(self) -> Iterator[Tuple[Square, Piece]]:
        """Every occupied square, as ``(square, piece)`` pairs.

        Scanning all 64 squares is wasteful compared to keeping an incrementally
        maintained piece list, but a piece list is another copy of the truth that
        has to be updated on every make and undo. At this scale the scan is
        free, and it cannot go stale.
        """
        for row in range(8):
            for col in range(8):
                piece = self.squares[row][col]
                if piece is not None:
                    yield (row, col), piece

    def pieces_of(self, color: str) -> Iterator[Tuple[Square, Piece]]:
        for square, piece in self.all_pieces():
            if piece.color == color:
                yield square, piece

    def king_square(self, color: str) -> Optional[Square]:
        for square, piece in self.pieces_of(color):
            if isinstance(piece, King):
                return square
        return None

    def is_attacked(self, square: Square, by_color: str) -> bool:
        """Does any piece of ``by_color`` bear down on ``square``?

        Brute force: ask every enemy piece what it attacks. The smarter approach
        is to look *outwards from the square* for each attacker type, which is
        far faster, but this version has one big advantage -- it reuses the exact
        same movement code that generates moves, so an attack map can never
        disagree with the move generator. One source of truth per rule.
        """
        for origin, piece in self.pieces_of(by_color):
            for target in piece.attacked_squares(self, origin):
                if target == square:
                    return True
        return False

    def in_check(self, color: str) -> bool:
        king = self.king_square(color)
        # The None guard exists only so contrived/test positions without a king
        # do not crash the generator.
        return king is not None and self.is_attacked(king, opponent(color))

    # ------------------------------------------------------- move generation
    def pseudo_legal_moves(self, color: Optional[str] = None) -> Iterator[Move]:
        """Every move that respects piece movement, king safety aside."""
        color = color or self.turn
        # pieces_of is materialised with list() because generate_moves may query
        # the board while we iterate; snapshotting avoids any chance of a
        # generator observing a half-mutated grid.
        for square, piece in list(self.pieces_of(color)):
            yield from piece.generate_moves(self, square)

    def legal_moves(self, color: Optional[str] = None) -> List[Move]:
        """Pseudo-legal moves minus any that leave the mover's own king in check.

        This is the make / test / unmake pattern described in the module
        docstring, and it is the single most important function in the engine.
        Because the test is "after playing this, is my king attacked?", it
        automatically handles absolute pins, discovered checks, kings stepping
        along an attacker's ray, and en passant captures that expose the king
        along a rank -- none of which need any dedicated code.

        Returns a list rather than an iterator on purpose: the board is mutated
        during generation, so a lazy iterator would interleave make/undo calls
        with whatever the caller does between iterations and corrupt the board.
        """
        color = color or self.turn
        legal = []
        for move in list(self.pseudo_legal_moves(color)):
            self.make_move(move)
            if not self.in_check(color):
                legal.append(move)
            self.undo_move()
        return legal

    def find_move(
        self, start: Square, end: Square, promotion: Optional[str] = None
    ) -> Optional[Move]:
        """Look up the fully-populated legal move matching a from/to request.

        Input layers should never construct a Move themselves; they describe the
        squares the user clicked or typed and let the generator hand back the
        real move, complete with its castle/en passant/promotion flags.
        """
        for move in self.legal_moves():
            if move.start == start and move.end == end:
                if promotion is None or move.promotion == promotion:
                    return move
        return None

    # ------------------------------------------------------- move execution
    def make_move(self, move: Move) -> None:
        """Play ``move``, pushing an undo record first.

        Assumes the move is at least pseudo-legal; validation is the caller's
        job. This runs for every candidate move during generation, so it stays
        free of checks it does not need.
        """
        piece = self.piece_at(move.start)
        if piece is None:
            raise ValueError(f"No piece on {square_name(move.start)}")

        # For en passant the captured pawn is beside the moving pawn, not on the
        # destination: same rank as the start, same file as the end.
        captured_square = move.end
        if move.is_en_passant:
            captured_square = (move.start[0], move.end[1])
        captured = self.piece_at(captured_square)

        rook = self.piece_at(move.rook_start) if move.is_castle else None
        # Snapshot before any mutation, so a failure mid-update cannot leave the
        # stack describing a state that never existed.
        self.undo_stack.append(
            _UndoState(
                move=move,
                piece=piece,
                captured=captured,
                captured_square=captured_square,
                piece_had_moved=piece.has_moved,
                rook_had_moved=rook.has_moved if rook else None,
                en_passant_target=self.en_passant_target,
                halfmove_clock=self.halfmove_clock,
                fullmove_number=self.fullmove_number,
                turn=self.turn,
            )
        )

        # Clear the victim first: for a normal capture this is the same square
        # as move.end and would be overwritten anyway, but for en passant it is
        # a different square that nothing else touches.
        self.set_piece(captured_square, None)
        self.set_piece(move.start, None)
        if move.promotion:
            # The pawn leaves the board entirely and a new piece arrives. The
            # pawn object survives in the undo record so it can come back.
            placed: Piece = PROMOTION_CLASSES[move.promotion](piece.color)
            placed.has_moved = True
        else:
            placed = piece
        self.set_piece(move.end, placed)
        piece.has_moved = True

        if move.is_castle and rook is not None:
            self.set_piece(move.rook_start, None)
            self.set_piece(move.rook_end, rook)
            rook.has_moved = True

        # En passant is only available on the immediately following ply, so this
        # is set on a double push and unconditionally cleared otherwise.
        if move.is_double_push:
            self.en_passant_target = ((move.start[0] + move.end[0]) // 2, move.start[1])
        else:
            self.en_passant_target = None

        # Pawn moves and captures are irreversible, so they reset the clock that
        # drives the fifty-move rule.
        if isinstance(piece, Pawn) or captured is not None:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        if self.turn == BLACK:
            self.fullmove_number += 1
        self.turn = opponent(self.turn)

    def undo_move(self) -> Optional[Move]:
        """Rewind the most recent move; returns it, or None if there was none.

        Every mutation performed by ``make_move`` is reversed here, and the two
        must be kept in lockstep. A missed field here does not cause a visible
        error -- it silently corrupts move generation, because legality testing
        undoes thousands of moves the player never sees.
        """
        if not self.undo_stack:
            return None
        state = self.undo_stack.pop()
        move = state.move

        # Order matters here. Clearing move.end before restoring the captured
        # piece is what makes a single code path work for both normal captures
        # (where captured_square == move.end) and en passant (where it does not).
        self.set_piece(move.end, None)
        self.set_piece(move.start, state.piece)
        # Restoring has_moved is essential, not cosmetic: without it, a castling
        # move merely *considered* during legality testing would permanently
        # revoke that side's castling rights.
        state.piece.has_moved = state.piece_had_moved
        if state.captured is not None:
            self.set_piece(state.captured_square, state.captured)

        if move.is_castle:
            rook = self.piece_at(move.rook_end)
            self.set_piece(move.rook_end, None)
            self.set_piece(move.rook_start, rook)
            if rook is not None:
                rook.has_moved = bool(state.rook_had_moved)

        self.en_passant_target = state.en_passant_target
        self.halfmove_clock = state.halfmove_clock
        self.fullmove_number = state.fullmove_number
        self.turn = state.turn
        return move

    # ------------------------------------------------------------ game state
    def is_checkmate(self, color: Optional[str] = None) -> bool:
        """In check with no legal reply.

        Checkmate and stalemate differ only in whether the king is currently
        attacked, which is why both reduce to the same "no legal moves" test.
        """
        color = color or self.turn
        return self.in_check(color) and not self.legal_moves(color)

    def is_stalemate(self, color: Optional[str] = None) -> bool:
        """Not in check, but no legal move exists -- a draw."""
        color = color or self.turn
        return not self.in_check(color) and not self.legal_moves(color)

    def has_insufficient_material(self) -> bool:
        """True for positions where checkmate is impossible for either side.

        Only the positions FIDE treats as an immediate draw are reported:
        K vs K, K+minor vs K, and K+B vs K+B with both bishops on the same
        colour squares. Positions that are merely *hard* to win (K+N+N vs K,
        say) are excluded, because mate there is possible with cooperation.

        The early return on pawns, rooks and queens is both a fast path and the
        rule itself: any of those pieces makes mate possible.
        """
        bishops: List[Tuple[Square, Piece]] = []
        knights = 0
        for square, piece in self.all_pieces():
            if isinstance(piece, (Pawn, Rook, Queen)):
                return False
            if isinstance(piece, Bishop):
                bishops.append((square, piece))
            elif isinstance(piece, Knight):
                knights += 1
        minor_count = len(bishops) + knights
        if minor_count <= 1:
            return True
        if knights == 0 and len(bishops) == 2:
            (sq_a, piece_a), (sq_b, piece_b) = bishops
            # (row + col) parity identifies a square's colour: bishops on the
            # same parity can never attack each other's squares, so neither side
            # can ever cover the squares needed to force mate.
            same_square_color = (sum(sq_a) % 2) == (sum(sq_b) % 2)
            if piece_a.color != piece_b.color and same_square_color:
                return True
        return False

    def position_key(self) -> str:
        """A string that uniquely identifies a position for repetition counting.

        Two positions count as "the same" for the threefold rule only if the
        piece placement, the side to move, the castling rights *and* the en
        passant possibility all match -- which is exactly what this encodes.
        Comparing piece placement alone would wrongly call positions equal when
        the available moves differ.

        A string is used rather than a hash because it cannot collide, and the
        cost is irrelevant at one call per move. A real engine would use Zobrist
        hashing here.
        """
        rows = []
        for row in range(8):
            rows.append(
                "".join(
                    piece.symbol if (piece := self.squares[row][col]) else "."
                    for col in range(8)
                )
            )
        castling = "".join(
            flag
            for flag, king_sq, rook_sq in (
                ("K", (0, 4), (0, 7)),
                ("Q", (0, 4), (0, 0)),
                ("k", (7, 4), (7, 7)),
                ("q", (7, 4), (7, 0)),
            )
            if self._castle_available(king_sq, rook_sq)
        )
        ep = square_name(self.en_passant_target) if self.en_passant_target else "-"
        return "/".join(rows) + f" {self.turn} {castling or '-'} {ep}"

    def _castle_available(self, king_sq: Square, rook_sq: Square) -> bool:
        """Whether a castling right still exists, for repetition keys only.

        This asks a narrower question than ``King._castling_moves``: it ignores
        blocking pieces and attacked squares, because the repetition rule cares
        about the *right* to castle, not whether it is playable right now.
        """
        king = self.piece_at(king_sq)
        rook = self.piece_at(rook_sq)
        return (
            isinstance(king, King)
            and not king.has_moved
            and isinstance(rook, Rook)
            and not rook.has_moved
            and king.color == rook.color
        )

    # ------------------------------------------------------------- notation
    def san(self, move: Move) -> str:
        """Standard Algebraic Notation for a legal move in the current position.

        SAN is context-dependent -- ``Nf3`` only makes sense given the position,
        and may need to become ``Ngf3`` if another knight could also go there.
        That is why this lives on the board and must be called *before* the move
        is played.
        """
        piece = self.piece_at(move.start)
        if piece is None:
            return move.uci

        if move.is_castle:
            # Kingside is towards the h-file, i.e. an increasing column index.
            text = "O-O" if move.end[1] > move.start[1] else "O-O-O"
        else:
            is_capture = self.piece_at(move.end) is not None or move.is_en_passant
            if isinstance(piece, Pawn):
                # Pawn captures are written with the departure file, e.g. exd5.
                text = square_name(move.start)[0] + "x" if is_capture else ""
                text += square_name(move.end)
            else:
                text = piece.letter.upper()
                text += self._disambiguation(move, piece)
                if is_capture:
                    text += "x"
                text += square_name(move.end)
            if move.promotion:
                text += "=" + move.promotion.upper()

        # The +/# suffix describes the position *after* the move, so the only
        # way to know it is to play the move, look, and take it back. Safe to do
        # here precisely because undo is exact.
        self.make_move(move)
        if self.is_checkmate():
            text += "#"
        elif self.in_check(self.turn):
            text += "+"
        self.undo_move()
        return text

    def _disambiguation(self, move: Move, piece: Piece) -> str:
        """The extra file/rank SAN needs when two like pieces can reach a square.

        Rivals are found from the *legal* move list, not the pseudo-legal one:
        a knight that is pinned cannot actually go there, so it creates no
        ambiguity and must not force a disambiguator.

        SAN's preference order is file, then rank, then the whole square -- the
        shortest form that is still unambiguous.
        """
        rivals = [
            other.start
            for other in self.legal_moves(piece.color)
            if other.end == move.end
            and other.start != move.start
            and type(self.piece_at(other.start)) is type(piece)
        ]
        if not rivals:
            return ""
        name = square_name(move.start)
        if all(sq[1] != move.start[1] for sq in rivals):
            return name[0]  # file alone is enough
        if all(sq[0] != move.start[0] for sq in rivals):
            return name[1]  # rank alone is enough
        return name  # three or more rivals: spell out the square

    def __str__(self) -> str:
        return self.render()

    def render(self, unicode: bool = False, flipped: bool = False) -> str:
        """ASCII/unicode diagram of the position, used by the console UI.

        Iteration order is reversed relative to storage because rank 8 prints
        first, and flipping simply reverses both axes so Black sits at the
        bottom. Orientation is a display concern only -- the underlying grid is
        never rearranged.
        """
        rows = range(8) if flipped else range(7, -1, -1)
        cols = range(7, -1, -1) if flipped else range(8)
        lines = ["  +------------------------+"]
        for row in rows:
            cells = []
            for col in cols:
                piece = self.squares[row][col]
                if piece is None:
                    cells.append("." if (row + col) % 2 else " ")
                else:
                    cells.append(piece.glyph if unicode else piece.symbol)
            lines.append(f"{row + 1} | " + "  ".join(cells) + " |")
        lines.append("  +------------------------+")
        files = "  ".join("abcdefgh"[c] for c in cols)
        lines.append("    " + files)
        return "\n".join(lines)
