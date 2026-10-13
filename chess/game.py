"""Game orchestration: turn loop, result detection and the console interface.

This layer sits on top of the board and owns everything that is about a *game*
rather than a *position*: who the players are, what has been played, what has
been captured, and how the game ended.

The split is deliberate and is what let the pygame GUI be added later without
touching a line of rule logic. ``Board`` answers "what is legal here?";
``Game`` answers "what has happened and is it over?"; the UI layers answer "what
did the user ask for?". Nothing below ever reaches upwards.

Draw rules are split across two methods for a reason. FIDE makes some draws
automatic and others *claimable* by a player:

* ``_update_result`` applies the automatic ones (stalemate, dead position, and
  the 75-move / fivefold backstops that end a game with no claim needed).
* ``claimable_draw`` reports the optional ones (fifty-move, threefold), which
  only take effect if someone actually asks.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .board import Board
from .move import Move, parse_square, square_name
from .pieces import BLACK, WHITE, Piece

HELP_TEXT = """
Commands
  e2e4 / e2 e4      make a move (from-square to-square)
  e7e8q             promote (q, r, b or n)
  O-O  /  O-O-O     castle (or just move the king two squares)
  moves [square]    list legal moves, optionally for one square
  board             redraw the board
  history           show the move list
  undo              take back the last move
  flip              flip the board orientation
  unicode           toggle unicode / letter pieces
  draw              offer a draw or claim one (50-move / threefold)
  resign            resign the game
  help              show this text
  quit              exit
"""


class Player:
    """A named side. Trivial today, but it gives later features (clocks,
    ratings, an AI opponent) an obvious place to live."""

    def __init__(self, color: str, name: Optional[str] = None) -> None:
        self.color = color
        self.name = name or color.capitalize()

    def __str__(self) -> str:
        return self.name


class Game:
    """Holds a board, the players and the rules that end a game."""

    def __init__(self, white_name: str = "White", black_name: str = "Black") -> None:
        self.board = Board()
        self.players: Dict[str, Player] = {
            WHITE: Player(WHITE, white_name),
            BLACK: Player(BLACK, black_name),
        }
        self.history: List[str] = []  # SAN strings, one per ply
        self.captured: Dict[str, List[Piece]] = {WHITE: [], BLACK: []}
        # None while the game is in progress; "1-0", "0-1", "1/2-1/2" or "*".
        self.result: Optional[str] = None
        self.reason: str = ""
        # Repetition counter. The starting position counts as its first
        # occurrence, which is why it is seeded here rather than left empty.
        self.position_counts: Dict[str, int] = {self.board.position_key(): 1}

    # ----------------------------------------------------------------- state
    @property
    def current_player(self) -> Player:
        return self.players[self.board.turn]

    @property
    def is_over(self) -> bool:
        return self.result is not None

    def play_move(self, move: Move) -> str:
        """Apply a legal move, record it and update the game result.

        Order is load-bearing: the victim must be read and the SAN generated
        *before* ``make_move``, because both describe the position as it is now.
        Afterwards the captured piece is gone and SAN could no longer be built.
        """
        victim = self.board.piece_at(
            # En passant takes a pawn that is not on the destination square.
            (move.start[0], move.end[1]) if move.is_en_passant else move.end
        )
        mover = self.board.turn
        notation = self.board.san(move)
        self.board.make_move(move)
        if victim is not None:
            self.captured[mover].append(victim)
        self.history.append(notation)
        key = self.board.position_key()
        self.position_counts[key] = self.position_counts.get(key, 0) + 1
        self._update_result()
        return notation

    def undo(self) -> bool:
        """Take back one ply, including this layer's own bookkeeping.

        The undo record is read *before* rewinding the board, because that is
        the only reliable way to know whether the move captured anything and who
        made it. An earlier version inferred the capture from an 'x' in the SAN
        string, which worked but tied game state to a display format.
        """
        if not self.board.undo_stack:
            return False
        state = self.board.undo_stack[-1]
        key = self.board.position_key()
        self.board.undo_move()
        self.position_counts[key] = max(0, self.position_counts.get(key, 1) - 1)
        self.history.pop()
        if state.captured is not None and self.captured[state.turn]:
            self.captured[state.turn].pop()
        # Taking back the move that ended the game resumes play.
        self.result = None
        self.reason = ""
        return True

    def _update_result(self) -> None:
        """Set result/reason if the position just played ended the game.

        Only the automatic endings are handled here; see the module docstring
        for why fifty-move and threefold are not in this list. The 75-move and
        fivefold variants *are*, because FIDE makes those mandatory -- they exist
        so a game cannot continue forever if neither player claims.
        """
        color = self.board.turn  # the side who must now move
        if self.board.is_checkmate(color):
            # The side to move is the one who lost.
            self.result = "0-1" if color == WHITE else "1-0"
            self.reason = f"Checkmate. {self.players[self._other(color)]} wins."
        elif self.board.is_stalemate(color):
            self.result = "1/2-1/2"
            self.reason = "Stalemate."
        elif self.board.has_insufficient_material():
            self.result = "1/2-1/2"
            self.reason = "Draw by insufficient material."
        elif self.board.halfmove_clock >= 150:
            # 150 plies = 75 full moves by each side.
            self.result = "1/2-1/2"
            self.reason = "Draw by the seventy-five move rule."
        elif max(self.position_counts.values(), default=0) >= 5:
            self.result = "1/2-1/2"
            self.reason = "Draw by fivefold repetition."

    @staticmethod
    def _other(color: str) -> str:
        return BLACK if color == WHITE else WHITE

    def claimable_draw(self) -> Optional[str]:
        """The name of a draw a player could claim right now, if any.

        Returns a description rather than a bool so the UI can tell the player
        exactly what they are claiming.
        """
        if self.board.halfmove_clock >= 100:  # 50 moves each
            return "the fifty-move rule"
        if self.position_counts.get(self.board.position_key(), 0) >= 3:
            return "threefold repetition"
        return None

    def resign(self, color: str) -> None:
        self.result = "0-1" if color == WHITE else "1-0"
        self.reason = f"{self.players[color]} resigned."

    def agree_draw(self, reason: str = "Draw agreed.") -> None:
        self.result = "1/2-1/2"
        self.reason = reason

    def formatted_history(self) -> str:
        """The move list as numbered pairs, e.g. ``1. e4  e5``.

        History is stored per ply but displayed per move, so the pairing happens
        here rather than at record time -- storing pairs would make undo of a
        single ply awkward.
        """
        if not self.history:
            return "No moves yet."
        lines = []
        for index in range(0, len(self.history), 2):
            pair = self.history[index : index + 2]
            number = index // 2 + 1
            lines.append(f"{number:>3}. {pair[0]:<8}" + (pair[1] if len(pair) > 1 else ""))
        return "\n".join(lines)


class ConsoleUI:
    """Text front-end driving a :class:`Game`.

    Built before the GUI, and kept afterwards: it is much faster to exercise
    rules through than a window, and having two independent front-ends is what
    proves the game layer has no hidden UI assumptions in it.
    """

    def __init__(self, game: Optional[Game] = None) -> None:
        self.game = game or Game()
        self.unicode = True
        self.flipped = False

    # ------------------------------------------------------------- rendering
    def show_board(self) -> None:
        board = self.game.board
        print()
        try:
            print(board.render(unicode=self.unicode, flipped=self.flipped))
        except UnicodeEncodeError:
            # Some Windows consoles cannot encode the chess glyphs. Rather than
            # detect that up front, try it and permanently fall back to letters.
            self.unicode = False
            print(board.render(unicode=False, flipped=self.flipped))
        taken_by_white = "".join(p.symbol for p in self.game.captured[WHITE])
        taken_by_black = "".join(p.symbol for p in self.game.captured[BLACK])
        if taken_by_white or taken_by_black:
            print(f"    captured -> White: {taken_by_white or '-'}   Black: {taken_by_black or '-'}")
        print()

    def status_line(self) -> str:
        board = self.game.board
        player = self.game.current_player
        check = " (check!)" if board.in_check(board.turn) else ""
        return f"Move {board.fullmove_number} - {player} to play{check}"

    # ------------------------------------------------------------- main loop
    def run(self) -> None:
        """Read-eval-print loop: commands first, otherwise treat input as a move."""
        print("=" * 46)
        print(" Two-player chess.  Type 'help' for commands.")
        print("=" * 46)
        self.show_board()

        while not self.game.is_over:
            print(self.status_line())
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                # Ctrl+C / Ctrl+D should exit cleanly, not dump a traceback.
                print("\nGoodbye.")
                return
            if not raw:
                continue
            if self.handle_command(raw):
                continue
            self.attempt_move(raw)

        self.show_board()
        print(f"Result: {self.game.result} - {self.game.reason}")
        print(self.game.formatted_history())

    def handle_command(self, raw: str) -> bool:
        """Return True when the input was a command rather than a move.

        The boolean return is what keeps commands and moves in one input field:
        anything unrecognised falls through to the move parser, so the player
        never has to prefix moves with anything.
        """
        command, _, argument = raw.lower().partition(" ")
        argument = argument.strip()

        if command in ("help", "?"):
            print(HELP_TEXT)
        elif command in ("quit", "exit"):
            self.game.result = "*"
            self.game.reason = "Game abandoned."
        elif command == "board":
            self.show_board()
        elif command == "history":
            print(self.game.formatted_history())
        elif command == "flip":
            self.flipped = not self.flipped
            self.show_board()
        elif command == "unicode":
            self.unicode = not self.unicode
            self.show_board()
        elif command == "undo":
            if self.game.undo():
                print("Move taken back.")
                self.show_board()
            else:
                print("Nothing to undo.")
        elif command == "moves":
            self.show_moves(argument)
        elif command == "resign":
            self.game.resign(self.game.board.turn)
        elif command == "draw":
            self.handle_draw()
        else:
            return False
        return True

    def show_moves(self, argument: str) -> None:
        """List legal moves, optionally filtered to one starting square.

        Mostly a debugging aid that turned out to be useful for players too:
        if a move is rejected, this shows exactly what was allowed instead.
        """
        board = self.game.board
        moves = board.legal_moves()
        if argument:
            try:
                square = parse_square(argument)
            except ValueError as error:
                print(error)
                return
            moves = [m for m in moves if m.start == square]
        if not moves:
            print("No legal moves.")
            return
        listing = sorted(f"{board.san(m)} ({m.uci})" for m in moves)
        for index in range(0, len(listing), 4):
            print("  " + "   ".join(f"{text:<16}" for text in listing[index : index + 4]))
        print(f"  {len(moves)} legal move(s).")

    def handle_draw(self) -> None:
        """Claim a draw if one is available, otherwise offer one.

        A claimable draw is unilateral, so it is applied immediately. An offer
        is not, so the opponent is prompted -- which works here only because
        both players share the keyboard.
        """
        claim = self.game.claimable_draw()
        if claim:
            self.game.agree_draw(f"Draw claimed by {claim}.")
            return
        opponent_player = self.game.players[Game._other(self.game.board.turn)]
        answer = input(f"{opponent_player}, accept the draw? (y/n) ").strip().lower()
        if answer.startswith("y"):
            self.game.agree_draw()
        else:
            print("Draw declined.")

    # ---------------------------------------------------------- move parsing
    def attempt_move(self, raw: str) -> None:
        """Parse and play, turning any parse failure into a readable message.

        ValueError is used as the single "bad input" channel so every rejection
        reason -- bad square, wrong colour, illegal move, missing promotion --
        surfaces the same way without a cascade of return codes.
        """
        try:
            move = self.parse_move(raw)
        except ValueError as error:
            print(f"{error}  (type 'help' for commands)")
            return
        notation = self.game.play_move(move)
        print(f"Played {notation}")
        self.show_board()

    def parse_move(self, raw: str) -> Move:
        """Turn typed text into a legal Move, or raise ValueError explaining why not.

        Accepts ``e2e4``, ``e2 e4``, ``e7e8q`` and the castling spellings. The
        strategy throughout is to never *construct* a move: the text is reduced
        to a from/to pair, and the answer is then looked up in the generated
        legal move list. That way this parser can never invent an illegal move
        or mislabel a special one.
        """
        board = self.game.board
        text = raw.replace("-", "").replace(",", " ").strip()

        # Castling first, since "O-O" does not name squares at all. Zeros are
        # folded to letters because both spellings are in common use.
        castle = raw.strip().lower().replace("0", "o")
        if castle in ("oo", "o-o", "o o"):
            return self._castle_move(long_side=False)
        if castle in ("ooo", "o-o-o", "o o o"):
            return self._castle_move(long_side=True)

        tokens = text.split()
        if len(tokens) == 1:
            token = tokens[0]
            if len(token) not in (4, 5):  # e2e4, or e7e8q with a promotion
                raise ValueError("Enter moves as 'e2e4' or 'e2 e4'.")
            squares, promotion = token[:4], token[4:].lower()
        elif len(tokens) == 2:
            squares = tokens[0] + tokens[1][:2]
            promotion = tokens[1][2:].lower()
        else:
            raise ValueError("Enter moves as 'e2e4' or 'e2 e4'.")

        start = parse_square(squares[:2])
        end = parse_square(squares[2:4])
        promotion = promotion or None
        if promotion and promotion not in "qrbn":
            raise ValueError("Promotion piece must be q, r, b or n.")

        # These two checks exist only to produce a better error message; the
        # legal move lookup below would reject them anyway.
        piece = board.piece_at(start)
        if piece is None:
            raise ValueError(f"There is no piece on {square_name(start)}.")
        if piece.color != board.turn:
            raise ValueError(f"The piece on {square_name(start)} is not yours.")

        candidates = [m for m in board.legal_moves() if m.start == start and m.end == end]
        if not candidates:
            raise ValueError(f"{square_name(start)}{square_name(end)} is not a legal move.")
        # More than one move for the same from/to pair can only mean promotion,
        # where the four choices differ solely by the piece created.
        if len(candidates) > 1:
            if promotion is None:
                raise ValueError("Promotion required: add q, r, b or n (e.g. e7e8q).")
            for move in candidates:
                if move.promotion == promotion:
                    return move
            raise ValueError("Promotion piece must be q, r, b or n.")
        return candidates[0]

    def _castle_move(self, long_side: bool) -> Move:
        """Find the castling move for this side, if it is currently legal.

        Asking the generator rather than building the move means all five
        castling conditions are already enforced -- there is nothing to re-check.
        """
        board = self.game.board
        for move in board.legal_moves():
            if not move.is_castle:
                continue
            # Queenside travel is towards the a-file, i.e. a lower column index.
            going_left = move.end[1] < move.start[1]
            if going_left == long_side:
                return move
        raise ValueError("Castling is not available.")


def main() -> None:
    """Console entry point; see main.py for the ``--console`` switch."""
    white = input("White player's name (Enter for 'White'): ").strip() or "White"
    black = input("Black player's name (Enter for 'Black'): ").strip() or "Black"
    ConsoleUI(Game(white, black)).run()
