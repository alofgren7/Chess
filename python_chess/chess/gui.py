"""Pygame front-end: a top-down board both players share with one mouse.

This is a pure presentation layer. It never decides whether a move is legal --
it asks ``Board.legal_moves()`` and filters that list by what the player
clicked. Every rule stays in the engine, which is why this file can be deleted
or replaced without affecting correctness.

Two ideas do most of the work here:

**Everything is derived from one Layout.** No pixel position is hardcoded twice.
``Layout.for_window`` computes the square size, board rect and sidebar rect from
the current window dimensions, and every draw and hit-test reads from it. That
is what makes resizing a five-line feature instead of a rewrite.

**Selection is a filter, not a state machine.** Clicking a piece stores its
square and the subset of legal moves starting there. Clicking again matches the
destination against that subset. There is no "awaiting destination" mode to get
stuck in -- any click that does not match simply reselects or clears.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pygame

from .game import Game
from .move import Move, Square
from .pieces import BLACK, PROMOTION_CLASSES, WHITE, Piece

MARGIN = 28
MIN_SQUARE = 40  # Below this, pieces stop being readable
LABEL_SPACE = 24  # Room under the board for the a-h file letters
DEFAULT_SQUARE = 80
MIN_SIDEBAR = 260
MAX_SIDEBAR = 380
# Minimums are derived from the same numbers the layout uses, so the window can
# never be resized into a state the layout cannot satisfy.
MIN_WIDTH = MIN_SQUARE * 8 + MARGIN * 3 + MIN_SIDEBAR
MIN_HEIGHT = MIN_SQUARE * 8 + MARGIN * 2 + LABEL_SPACE
DEFAULT_WIDTH = DEFAULT_SQUARE * 8 + MARGIN * 3 + 300
DEFAULT_HEIGHT = DEFAULT_SQUARE * 8 + MARGIN * 2 + LABEL_SPACE
FPS = 60

# Turn announcement timing, in milliseconds.
FLASH_MS = 1600
FLASH_FADE_IN = 120
FLASH_FADE_OUT = 450

# Fonts known to carry the U+2654-265F chess glyphs, best first. Tried in order
# because no single one of these is present on every platform.
GLYPH_FONTS = (
    "segoeuisymbol",
    "dejavusans",
    "freeserif",
    "arialunicodems",
    "symbola",
    "notosanssymbols2",
)


@dataclass(frozen=True)
class Layout:
    """Board and sidebar geometry derived from the current window size.

    Frozen on purpose: a Layout is a snapshot for one window size, not a mutable
    object to poke at. Resizing builds a new one, so there is no chance of half
    the frame being drawn with old coordinates and half with new ones.
    """

    square: int
    board: pygame.Rect
    sidebar: pygame.Rect

    @classmethod
    def for_window(cls, width: int, height: int) -> "Layout":
        """Fit the largest square board that leaves room for the sidebar.

        The sidebar takes a proportion of the width but is clamped, so it never
        squeezes the board on small windows or wastes half the screen on large
        ones. The board square size is then whichever of the two axes is the
        binding constraint, floored at MIN_SQUARE.

        Squares are integers so the board never lands on fractional pixels,
        which would leave visible seams between squares.
        """
        sidebar_width = min(max(MIN_SIDEBAR, int(width * 0.26)), MAX_SIDEBAR)
        usable_width = width - MARGIN * 3 - sidebar_width
        usable_height = height - MARGIN * 2 - LABEL_SPACE
        square = max(MIN_SQUARE, min(usable_width // 8, usable_height // 8))
        size = square * 8
        # Centre vertically in the leftover space so a wide window does not pin
        # the board to the top edge.
        top = MARGIN + max(0, (usable_height - size) // 2)
        board = pygame.Rect(MARGIN, top, size, size)
        # The sidebar claims whatever is left, rather than its nominal width, so
        # rounding in the board size never leaves a gap at the right edge.
        sidebar = pygame.Rect(
            board.right + MARGIN, MARGIN, width - board.right - MARGIN * 2, height - MARGIN * 2
        )
        return cls(square, board, sidebar)

    @property
    def piece_size(self) -> int:
        """Glyph point size; 78% of a square leaves a visible border of board."""
        return int(self.square * 0.78)


@dataclass(frozen=True)
class Theme:
    """Every colour in one place, so the look can be changed without hunting
    through draw calls for literals."""

    light: Tuple[int, int, int] = (238, 222, 189)
    dark: Tuple[int, int, int] = (152, 110, 77)
    background: Tuple[int, int, int] = (32, 30, 28)
    panel: Tuple[int, int, int] = (45, 42, 39)
    text: Tuple[int, int, int] = (236, 232, 226)
    muted: Tuple[int, int, int] = (150, 144, 136)
    white_piece: Tuple[int, int, int] = (250, 250, 248)
    black_piece: Tuple[int, int, int] = (28, 26, 24)
    selection: Tuple[int, int, int] = (246, 202, 78)
    last_move: Tuple[int, int, int] = (205, 190, 96)
    check: Tuple[int, int, int] = (212, 76, 62)
    hint: Tuple[int, int, int] = (40, 40, 40)
    button: Tuple[int, int, int] = (62, 58, 54)
    button_hover: Tuple[int, int, int] = (84, 79, 73)
    accent: Tuple[int, int, int] = (128, 176, 132)


class PieceRenderer:
    """Draws pieces as outlined unicode glyphs, with a letter fallback.

    Using font glyphs avoids shipping image assets entirely, at the cost of
    depending on a font that actually contains them -- hence the detection and
    fallback below.

    Note that the *filled* (nominally black) glyphs are used for both colours,
    recoloured and outlined. The outlined U+2654-2659 variants render as thin
    line art that reads poorly at small sizes and can vanish against a light
    square, whereas a filled shape with a contrasting outline stays legible on
    any background and at any size.
    """

    GLYPHS = {"K": "\u265a", "Q": "\u265b", "R": "\u265c", "B": "\u265d", "N": "\u265e", "P": "\u265f"}

    def __init__(self, theme: Theme) -> None:
        self.theme = theme
        # Rendering text is expensive and the same twelve pieces are drawn every
        # frame, so finished surfaces are cached by (colour, piece, size). The
        # size is part of the key because resizing needs a fresh set.
        self._cache: Dict[Tuple[str, str, int], pygame.Surface] = {}
        self._glyph_fonts: Dict[int, Optional[pygame.font.Font]] = {}
        self._letter_fonts: Dict[int, pygame.font.Font] = {}

    def _glyph_font(self, size: int) -> Optional[pygame.font.Font]:
        """First installed font that really has the chess glyphs, or None.

        ``metrics()`` returns None for characters a font does not contain, which
        is the only reliable way to detect this: a missing glyph still *renders*
        (as an empty box) and still reports a width, so measuring is not enough.
        """
        if size not in self._glyph_fonts:
            chosen = None
            for name in GLYPH_FONTS:
                path = pygame.font.match_font(name)
                if not path:
                    continue
                candidate = pygame.font.Font(path, size)
                if candidate.metrics("\u265a")[0] is not None:
                    chosen = candidate
                    break
            self._glyph_fonts[size] = chosen
        return self._glyph_fonts[size]

    def _letter_font(self, size: int) -> pygame.font.Font:
        """Fallback face for drawing K/Q/R/B/N/P when no glyph font exists."""
        if size not in self._letter_fonts:
            self._letter_fonts[size] = pygame.font.SysFont("georgia,timesnewroman,serif", size, bold=True)
        return self._letter_fonts[size]

    def surface(self, piece: Piece, size: int) -> pygame.Surface:
        key = (piece.color, piece.letter, size)
        if key not in self._cache:
            self._cache[key] = self._render(piece, size)
        return self._cache[key]

    def _render(self, piece: Piece, size: int) -> pygame.Surface:
        """Draw the glyph in its fill colour over a contrasting outline.

        The outline is faked by blitting the same glyph in the opposite colour
        at eight surrounding offsets and then drawing the fill on top. Crude
        compared to a real stroked font, but it needs no extra assets and is
        what keeps white pieces visible on light squares and black on dark.
        """
        fill = self.theme.white_piece if piece.color == WHITE else self.theme.black_piece
        outline = self.theme.black_piece if piece.color == WHITE else self.theme.white_piece
        font = self._glyph_font(size)
        if font is not None:
            text = self.GLYPHS[piece.letter]
        else:
            font = self._letter_font(size)
            text = piece.letter

        body = font.render(text, True, fill)
        edge = font.render(text, True, outline)
        thickness = max(1, size // 32)  # scales with the board
        # Canvas is padded by the outline width so the offsets are not clipped.
        canvas = pygame.Surface(
            (body.get_width() + thickness * 2, body.get_height() + thickness * 2),
            pygame.SRCALPHA,
        )
        for d_x in (-thickness, 0, thickness):
            for d_y in (-thickness, 0, thickness):
                if d_x or d_y:
                    canvas.blit(edge, (thickness + d_x, thickness + d_y))
        canvas.blit(body, (thickness, thickness))
        return canvas


class Button:
    """A labelled rectangle that reports an action string when clicked.

    It holds no callback. The owner matches on ``action`` instead, which keeps
    all behaviour in one readable dispatch method rather than scattered across
    lambdas, and lets a button be rebuilt on resize without rebinding anything.
    """

    def __init__(self, label: str, rect: pygame.Rect, action: str) -> None:
        self.label = label
        self.rect = rect
        self.action = action
        self.enabled = True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, theme: Theme, mouse: Tuple[int, int]) -> None:
        # Hover is computed at draw time from the live mouse position rather
        # than tracked as state, so it can never get stuck after a resize or
        # when the pointer leaves the window.
        hovered = self.enabled and self.rect.collidepoint(mouse)
        color = theme.button_hover if hovered else theme.button
        if not self.enabled:
            color = theme.panel  # blends into the sidebar: clearly inert
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, theme.muted if self.enabled else theme.panel, self.rect, width=1, border_radius=6)
        text_color = theme.text if self.enabled else theme.muted
        label = font.render(self.label, True, text_color)
        surface.blit(label, label.get_rect(center=self.rect.center))


class ChessGUI:
    """Hot-seat chess: both players click on the same board, turn by turn.

    Because both players share one mouse, there is no concept of "my pieces" --
    the only gate is ``board.turn``. Clicking a piece of the side not to move
    simply does nothing.
    """

    def __init__(
        self,
        game: Optional[Game] = None,
        theme: Optional[Theme] = None,
        screen: Optional[pygame.Surface] = None,
    ) -> None:
        # An existing surface can be passed in so the name screen and the board
        # share one window; the app loop hands the same surface to both.
        self.screen = screen or open_window()
        self.layout = Layout.for_window(*self.screen.get_size())
        self.clock = pygame.time.Clock()
        self.theme = theme or Theme()
        self.renderer = PieceRenderer(self.theme)

        # Sidebar fonts are fixed size: the sidebar itself is clamped in width,
        # so scaling them with the window would gain nothing. Only the board and
        # the flash banner scale.
        self.font_large = pygame.font.SysFont("segoeui,arial", 26, bold=True)
        self.font = pygame.font.SysFont("segoeui,arial", 19)
        self.font_small = pygame.font.SysFont("consolas,couriernew", 16)
        self.font_button = pygame.font.SysFont("segoeui,arial", 17, bold=True)
        self._flash_fonts: Dict[int, pygame.font.Font] = {}

        self.game = game or Game()
        self.flipped = False  # display orientation only; never touches the board
        self.auto_flip = False
        self.selected: Optional[Square] = None
        self.targets: List[Move] = []  # legal moves from `selected`
        # Recomputed once per position instead of per frame: legal_moves is the
        # expensive call in the whole program, and nothing changes between moves.
        self.legal_moves: List[Move] = []
        self.last_move: Optional[Move] = None
        # Non-empty while the promotion picker is open; holds the four candidate
        # moves, which is all the state that modal state needs.
        self.promotion_choices: List[Move] = []
        self.promotion_rects: List[Tuple[pygame.Rect, Move]] = []
        self.message = "White to move."
        self.confirming: Optional[str] = None  # action awaiting a second click
        self.flash_text = ""
        self.flash_color = self.theme.text
        self.flash_started = 0
        # How run() ended, so the app loop knows whether to reopen the name screen.
        self.exit_action = "quit"
        self.buttons = self._build_buttons()
        self._refresh_moves()
        self._flash_turn()

    # ------------------------------------------------------------- lifecycle
    def _build_buttons(self) -> List[Button]:
        """Lay the sidebar buttons out from the bottom up.

        Anchoring to the bottom means the variable-height content above (move
        history) grows into the space that is left, rather than pushing the
        controls off screen. Spacing shrinks on short windows so all six stay
        visible.

        Rebuilt on every resize, which is also why labels are reset here --
        a button left reading "Click again to confirm" would otherwise survive.
        """
        bar = self.layout.sidebar
        x = bar.left + 20
        width = bar.width - 40
        labels = [
            ("Undo  (U)", "undo"),
            ("Flip board  (F)", "flip"),
            ("New game  (N)", "new"),
            ("Change players  (P)", "rename"),
            ("Offer draw", "draw"),
            ("Resign", "resign"),
        ]
        spacing = max(26, min(40, bar.height // 11))
        top = bar.bottom - len(labels) * spacing
        buttons = [
            Button(label, pygame.Rect(x, top + index * spacing, width, spacing - 8), action)
            for index, (label, action) in enumerate(labels)
        ]
        for button in buttons:
            if button.action == "draw":
                button.label = "Offer draw"
            elif button.action == "resign":
                button.label = "Resign"
        return buttons

    def _resize(self, width: int, height: int) -> None:
        """Rebuild everything that depends on window size.

        Clamping happens here rather than being left to the window manager,
        because SDL will happily deliver a resize smaller than the layout can
        satisfy; calling set_mode with the clamped size pushes back.
        """
        width = max(MIN_WIDTH, width)
        height = max(MIN_HEIGHT, height)
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.layout = Layout.for_window(width, height)
        self.buttons = self._build_buttons()
        self._refresh_button_state()

    def run(self) -> str:
        """Play until the window closes; returns 'quit' or 'rename'.

        A standard immediate-mode loop: drain events, then redraw the whole
        frame. Nothing is cached between frames except the legal move list, so
        there is no invalidation logic to get wrong -- at 60fps a full redraw of
        64 squares is far cheaper than tracking dirty rectangles.
        """
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self._resize(event.w, event.h)
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        running = self._handle_click(event.pos)
                    elif event.button == 3:
                        self._clear_selection()
            self.draw()
            self.clock.tick(FPS)
        return self.exit_action

    # ------------------------------------------------------------------ flash
    def flash(self, text: str, color: Optional[Tuple[int, int, int]] = None) -> None:
        """Show a transient banner over the board.

        Implemented as a timestamp plus a string rather than an animation object
        or a timer event: the draw loop already runs every frame, so it can
        compute the fade from elapsed time. Nothing needs to be ticked, cancelled
        or cleaned up, and calling flash() again simply replaces the old one.
        """
        self.flash_text = text
        self.flash_color = color or self.theme.text
        self.flash_started = pygame.time.get_ticks()

    def _flash_turn(self) -> None:
        """Announce whose turn it now is, or why the game ended.

        Called after anything that changes the side to move. With both players
        on one screen this is the main defence against someone moving twice.
        """
        if self.game.is_over:
            self.flash(self.game.reason, self.theme.accent)
            return
        color = self.game.board.turn
        player = self.game.players[color]
        suffix = " - check!" if self.game.board.in_check(color) else ""
        side = "White" if color == WHITE else "Black"
        label = f"{player}'s turn ({side}){suffix}"
        self.flash(label, self.theme.check if suffix else self.theme.text)

    # ------------------------------------------------------------ game state
    def _refresh_moves(self) -> None:
        """Recache the legal moves and anything derived from them.

        Called once per position change, never per frame. Everything the UI
        needs to know about legality comes from this one cached list.
        """
        self.legal_moves = [] if self.game.is_over else self.game.board.legal_moves()
        self._update_message()
        self._refresh_button_state()

    def _refresh_button_state(self) -> None:
        # Split out from _refresh_moves because a resize rebuilds the buttons
        # and needs to re-apply their enabled state without recomputing moves.
        for button in self.buttons:
            if button.action in ("draw", "resign"):
                button.enabled = not self.game.is_over
            elif button.action == "undo":
                button.enabled = bool(self.game.board.undo_stack)

    def _update_message(self) -> None:
        """The persistent sidebar status line, as opposed to the transient flash."""
        if self.game.is_over:
            self.message = self.game.reason
            return
        board = self.game.board
        player = self.game.current_player
        if board.in_check(board.turn):
            self.message = f"{player} is in check!"
        else:
            self.message = f"{player} to move."

    def _clear_selection(self) -> None:
        """Drop all transient interaction state.

        Includes the pending confirm, so starting to resign and then clicking
        the board cancels the resignation rather than leaving it armed.
        """
        self.selected = None
        self.targets = []
        self.promotion_choices = []
        self.confirming = None

    def _new_game(self) -> None:
        # Replaces the Game outright rather than resetting it: fewer fields to
        # remember to clear, and no chance of stale history or counters.
        self.game = Game(self.game.players[WHITE].name, self.game.players[BLACK].name)
        self.flipped = False
        self.last_move = None
        self._clear_selection()
        self._refresh_moves()
        self._flash_turn()

    def _undo(self) -> None:
        if self.game.undo():
            # After undoing, the highlight should show the move before the one
            # just taken back, which is whatever is now on top of the stack.
            self.last_move = self.game.board.undo_stack[-1].move if self.game.board.undo_stack else None
            self._clear_selection()
            if self.auto_flip:
                self.flipped = self.game.board.turn == BLACK
            self._refresh_moves()
            self._flash_turn()

    def _apply_move(self, move: Move) -> None:
        """The single funnel every move passes through, whatever triggered it.

        Keeping board update, highlight, orientation, cache refresh and flash in
        one place means a new input path (a promotion click, a dragged piece, a
        future AI) cannot forget one of those steps.
        """
        self.game.play_move(move)
        self.last_move = move
        self._clear_selection()
        if self.auto_flip:
            self.flipped = self.game.board.turn == BLACK
        self._refresh_moves()
        self._flash_turn()

    # ----------------------------------------------------------------- input
    def _handle_key(self, key: int) -> bool:
        """Returns False to stop the event loop."""
        if key in (pygame.K_ESCAPE,):
            self._clear_selection()
        elif key == pygame.K_u:
            self._undo()
        elif key == pygame.K_f:
            self.flipped = not self.flipped
        elif key == pygame.K_a:
            self.auto_flip = not self.auto_flip
            if self.auto_flip:
                self.flipped = self.game.board.turn == BLACK
            self.flash(f"Auto-flip {'on' if self.auto_flip else 'off'}")
        elif key == pygame.K_n:
            self._new_game()
        elif key == pygame.K_p:
            # Leaving the loop is how the name screen is reached; run() reports
            # the reason to the app loop, which reopens it.
            self.exit_action = "rename"
            return False
        elif key == pygame.K_q:
            return False
        return True

    def _handle_click(self, position: Tuple[int, int]) -> bool:
        """Route a left click. Returns False to stop the event loop.

        Order is the whole design: the promotion picker is modal and swallows
        everything, then buttons, then the board. Without that precedence a
        click on the picker could fall through to the square behind it.
        """
        if self.promotion_choices:
            for rect, move in self.promotion_rects:
                if rect.collidepoint(position):
                    self._apply_move(move)
                    return True
            # Clicking outside the picker cancels rather than picking a default:
            # silently queening someone's pawn would be worse than doing nothing.
            self._clear_selection()
            return True

        for button in self.buttons:
            if button.enabled and button.rect.collidepoint(position):
                if button.action == "rename":
                    self.exit_action = "rename"
                    return False
                self._handle_button(button.action)
                return True

        square = self.square_at(position)
        if square is None or self.game.is_over:
            return True
        # Any board click abandons a pending resign/draw confirmation.
        self.confirming = None

        if self.selected is not None:
            # `targets` is already filtered to legal moves from the selected
            # square, so a match here needs no further validation.
            matches = [m for m in self.targets if m.end == square]
            if len(matches) == 1:
                self._apply_move(matches[0])
                return True
            if len(matches) > 1:
                # Several moves to the same square can only be a promotion;
                # storing them opens the picker on the next frame.
                self.promotion_choices = matches
                return True

        piece = self.game.board.piece_at(square)
        if piece is not None and piece.color == self.game.board.turn:
            self.selected = square
            self.targets = [m for m in self.legal_moves if m.start == square]
        else:
            self._clear_selection()
        return True

    def _handle_button(self, action: str) -> None:
        """Dispatch a sidebar button. 'rename' is handled by the caller, since
        it has to stop the event loop."""
        if action == "undo":
            self._undo()
        elif action == "flip":
            self.flipped = not self.flipped
        elif action == "new":
            self._new_game()
        elif action in ("draw", "resign"):
            self._handle_confirmable(action)

    def _handle_confirmable(self, action: str) -> None:
        """Two-click confirmation for the irreversible actions.

        Resigning or agreeing a draw ends the game, so a stray click should not
        do it. Rather than a modal dialog, the button itself becomes the
        confirmation: the first click relabels it, the second commits. That
        needs one extra field (``confirming``) and no new event handling, and
        any other click cancels it naturally via ``_clear_selection``.
        """
        button = next(b for b in self.buttons if b.action == action)
        if self.confirming != action:
            # Arming one confirmable disarms the other.
            self._reset_button_labels()
            self.confirming = action
            claim = self.game.claimable_draw()
            if action == "draw" and claim:
                # Name the rule, since claiming is a right rather than an offer.
                button.label = f"Claim draw ({claim})?"
            else:
                button.label = "Click again to confirm"
            return

        if action == "resign":
            self.game.resign(self.game.board.turn)
        else:
            claim = self.game.claimable_draw()
            self.game.agree_draw(f"Draw claimed by {claim}." if claim else "Draw agreed.")
        self._reset_button_labels()
        self._clear_selection()
        self._refresh_moves()
        self._flash_turn()

    def _reset_button_labels(self) -> None:
        for button in self.buttons:
            if button.action == "draw":
                button.label = "Offer draw"
            elif button.action == "resign":
                button.label = "Resign"
        self.confirming = None

    # ------------------------------------------------------------ geometry
    def square_at(self, position: Tuple[int, int]) -> Optional[Square]:
        """Pixel -> board square, honouring the current orientation.

        The exact inverse of ``square_rect``; the two must always agree, and a
        cheap way to verify that is to check the round trip
        ``square_at(square_rect(s).center) == s`` for both orientations.

        Board rows run bottom-up while screen rows run top-down, which is why
        the unflipped case is ``7 - row_index``. Flipping mirrors both axes.
        """
        board = self.layout.board
        if not board.collidepoint(position):
            return None
        col_index = (position[0] - board.left) // self.layout.square
        row_index = (position[1] - board.top) // self.layout.square
        if not (0 <= col_index < 8 and 0 <= row_index < 8):
            return None
        row = row_index if self.flipped else 7 - row_index
        col = 7 - col_index if self.flipped else col_index
        return (int(row), int(col))

    def square_rect(self, square: Square) -> pygame.Rect:
        """Board square -> the pixel rectangle it occupies right now.

        Every board-relative draw call goes through this, so orientation and
        window size are handled in exactly one place.
        """
        row, col = square
        board = self.layout.board
        size = self.layout.square
        col_index = 7 - col if self.flipped else col
        row_index = row if self.flipped else 7 - row
        return pygame.Rect(board.left + col_index * size, board.top + row_index * size, size, size)

    # ------------------------------------------------------------- rendering
    def draw(self) -> None:
        """Render one frame, back to front.

        Painter's algorithm: each layer may cover the ones before it. The flash
        banner is last so it floats above everything, and the promotion picker
        sits above the pieces but below the sidebar.
        """
        self.screen.fill(self.theme.background)
        self.draw_board()
        self.draw_coordinates()
        self.draw_pieces()
        if self.promotion_choices:
            self.draw_promotion()
        self.draw_sidebar()
        self.draw_flash()
        pygame.display.flip()

    def draw_board(self) -> None:
        """Squares, then highlights, then move hints.

        Highlights are drawn as translucent overlays rather than as alternative
        square colours, so they compose: a selected square that was also part of
        the last move shows both tints, and one theme colour works on light and
        dark squares alike.
        """
        board = self.game.board
        size = self.layout.square
        for row in range(8):
            for col in range(8):
                square = (row, col)
                rect = self.square_rect(square)
                # (row + col) parity gives the checkerboard; odd is light, which
                # correctly puts a dark square on a1.
                base = self.theme.light if (row + col) % 2 else self.theme.dark
                pygame.draw.rect(self.screen, base, rect)
                if self.last_move and square in (self.last_move.start, self.last_move.end):
                    self._tint(rect, self.theme.last_move, 110)
                if square == self.selected:
                    self._tint(rect, self.theme.selection, 140)

        # Suppress the check highlight on a drawn game, where the "side to move"
        # is not really under threat any more.
        if not self.game.is_over or self.game.result != "1/2-1/2":
            king_square = board.king_square(board.turn)
            if king_square and board.in_check(board.turn):
                self._tint(self.square_rect(king_square), self.theme.check, 150)

        for move in self.targets:
            rect = self.square_rect(move.end)
            # En passant destinations look empty but are captures, so they get
            # the ring rather than the dot.
            occupied = board.piece_at(move.end) is not None or move.is_en_passant
            overlay = pygame.Surface((size, size), pygame.SRCALPHA)
            center = (size // 2, size // 2)
            if occupied:
                # A ring around the victim, so the piece underneath stays visible.
                pygame.draw.circle(
                    overlay, (*self.theme.hint, 120), center, size // 2 - 3, max(3, size // 14)
                )
            else:
                pygame.draw.circle(overlay, (*self.theme.hint, 110), center, size // 6)
            self.screen.blit(overlay, rect.topleft)

        pygame.draw.rect(self.screen, self.theme.panel, self.layout.board.inflate(4, 4), width=2)

    def _tint(self, rect: pygame.Rect, color: Tuple[int, int, int], alpha: int) -> None:
        """Blend a translucent colour over a rectangle.

        pygame cannot draw a primitive with alpha directly onto the display
        surface, so the standard workaround is to build a small SRCALPHA
        surface, fill it, and blit that.
        """
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        overlay.fill((*color, alpha))
        self.screen.blit(overlay, rect.topleft)

    def draw_coordinates(self) -> None:
        """File letters below the board and rank numbers to its left.

        ``index`` walks screen positions while ``row``/``col`` are the board
        values those positions currently show -- the same mapping as
        ``square_rect``, so labels follow the board when it is flipped.
        """
        board = self.layout.board
        size = self.layout.square
        for index in range(8):
            col = 7 - index if self.flipped else index
            row = index if self.flipped else 7 - index
            file_label = self.font_small.render("abcdefgh"[col], True, self.theme.muted)
            rank_label = self.font_small.render(str(row + 1), True, self.theme.muted)
            self.screen.blit(
                file_label,
                (board.left + index * size + size // 2 - file_label.get_width() // 2, board.bottom + 6),
            )
            self.screen.blit(
                rank_label,
                (board.left - 18, board.top + index * size + size // 2 - rank_label.get_height() // 2),
            )

    def draw_pieces(self) -> None:
        # Centring the glyph rather than aligning it to the square's corner
        # keeps pieces optically centred despite the font's own side bearings.
        for square, piece in self.game.board.all_pieces():
            surface = self.renderer.surface(piece, self.layout.piece_size)
            self.screen.blit(surface, surface.get_rect(center=self.square_rect(square).center))

    def draw_promotion(self) -> None:
        """The four-choice picker, drawn as a column at the promotion square.

        Placing it on the destination file makes the connection to the pawn
        obvious without an explanatory label. The hit rectangles are recorded
        here, at draw time, rather than computed separately -- so what is clicked
        is by definition exactly what was drawn, even after a resize.
        """
        self.promotion_rects = []
        board = self.layout.board
        size = self.layout.square
        anchor = self.square_rect(self.promotion_choices[0].end)
        color = self.game.board.turn
        height = size * len(self.promotion_choices)
        # Clamp so the column stays on the board whichever end it promotes at.
        top = min(max(board.top, anchor.top), board.bottom - height)
        panel = pygame.Rect(anchor.left, top, size, height)

        # Dim the board behind to signal that this is modal.
        shade = pygame.Surface(board.size, pygame.SRCALPHA)
        shade.fill((0, 0, 0, 120))
        self.screen.blit(shade, board.topleft)
        pygame.draw.rect(self.screen, self.theme.panel, panel.inflate(8, 8), border_radius=8)

        # Fixed order with queen first: it is the overwhelmingly common choice,
        # and a stable order builds muscle memory.
        order = {"q": 0, "r": 1, "b": 2, "n": 3}
        choices = sorted(self.promotion_choices, key=lambda m: order.get(m.promotion or "q", 9))
        for index, move in enumerate(choices):
            cell = pygame.Rect(panel.left, panel.top + index * size, size, size)
            hovered = cell.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(self.screen, self.theme.light if hovered else (215, 205, 188), cell)
            sample = self._promotion_piece(move, color)
            surface = self.renderer.surface(sample, self.layout.piece_size)
            self.screen.blit(surface, surface.get_rect(center=cell.center))
            self.promotion_rects.append((cell, move))

    @staticmethod
    def _promotion_piece(move: Move, color: str) -> Piece:
        """A throwaway piece instance, created only so it can be drawn."""
        return PROMOTION_CLASSES[move.promotion or "q"](color)

    def draw_sidebar(self) -> None:
        """Title, players, status and move list, laid out top-down.

        Content is drawn until it would collide with the buttons (which are
        anchored to the bottom), and then stops. That ``limit`` check is what
        lets the window shrink to its minimum without text overlapping.
        """
        theme = self.theme
        bar = self.layout.sidebar
        pygame.draw.rect(self.screen, theme.panel, bar, border_radius=8)
        inner_x = bar.left + 20
        y = bar.top + 18
        limit = self.buttons[0].rect.top - 8

        title = self.font_large.render("Python Chess", True, theme.text)
        self.screen.blit(title, (inner_x, y))
        y += 40

        for color in (WHITE, BLACK):
            if y + 30 > limit:
                break  # no room left; the buttons matter more than the names
            player = self.game.players[color]
            active = not self.game.is_over and self.game.board.turn == color
            # A small swatch in the piece colour, so which row is which side is
            # obvious even when both players have custom names.
            dot = pygame.Rect(inner_x, y + 5, 12, 12)
            pygame.draw.rect(self.screen, theme.white_piece if color == WHITE else theme.black_piece, dot, border_radius=3)
            pygame.draw.rect(self.screen, theme.muted, dot, width=1, border_radius=3)
            label = self.font.render(str(player), True, theme.accent if active else theme.muted)
            self.screen.blit(label, (inner_x + 22, y))
            if active:
                # Drawn as a polygon rather than a text arrow: not every system
                # font has the triangle glyphs, and a missing one renders as a
                # box. Shapes always work.
                tip = inner_x + 30 + label.get_width()
                mid = y + label.get_height() // 2
                pygame.draw.polygon(
                    self.screen, theme.accent, [(tip, mid), (tip + 11, mid - 7), (tip + 11, mid + 7)]
                )
            taken = self.game.captured[color]
            if taken:
                # Most valuable first, and overlapped (17px step for a 22px
                # glyph) so a long row of captures still fits the sidebar.
                offset = inner_x + 22
                for piece in sorted(taken, key=lambda p: -p.points):
                    glyph = self.renderer.surface(piece, 22)
                    self.screen.blit(glyph, (offset, y + 24))
                    offset += 17
            y += 54

        advantage = self._material_advantage()
        if advantage and y - 8 + 18 <= limit:
            self.screen.blit(self.font_small.render(advantage, True, theme.muted), (inner_x, y - 8))
        y += 8

        if y + 24 <= limit:
            status = self.font.render(self.message, True, theme.text)
            self.screen.blit(status, (inner_x, y))
            y += 30
            if self.game.is_over and y + 24 <= limit:
                result = self.font.render(f"Result: {self.game.result}", True, theme.accent)
                self.screen.blit(result, (inner_x, y))
                y += 26

        hint_y = self.buttons[0].rect.top - 22
        if hint_y > y + 8:
            pygame.draw.line(self.screen, theme.button, (inner_x, y), (bar.right - 20, y))
            self._draw_history(inner_x, y + 12, hint_y - 6)
            hint = self.font_small.render("A: auto-flip   Esc: clear", True, theme.muted)
            self.screen.blit(hint, (inner_x, hint_y))

        mouse = pygame.mouse.get_pos()
        for button in self.buttons:
            button.draw(self.screen, self.font_button, theme, mouse)

    def _flash_font(self) -> pygame.font.Font:
        """Banner font scaled to the board, clamped so it stays readable but
        never dominates a large window."""
        size = max(20, min(40, self.layout.square // 2))
        if size not in self._flash_fonts:
            self._flash_fonts[size] = pygame.font.SysFont("segoeui,arial", size, bold=True)
        return self._flash_fonts[size]

    def draw_flash(self) -> None:
        """Draw the turn banner, fading in and out over its lifetime.

        The fade is recomputed from elapsed time on every frame, so it is
        frame-rate independent and self-expiring: once the banner is older than
        FLASH_MS the text is cleared and nothing is drawn again until the next
        flash() call.
        """
        if not self.flash_text:
            return
        elapsed = pygame.time.get_ticks() - self.flash_started
        if elapsed >= FLASH_MS:
            self.flash_text = ""
            return
        # Quick fade in, long hold, slower fade out.
        if elapsed < FLASH_FADE_IN:
            fade = elapsed / FLASH_FADE_IN
        elif elapsed > FLASH_MS - FLASH_FADE_OUT:
            fade = (FLASH_MS - elapsed) / FLASH_FADE_OUT
        else:
            fade = 1.0

        text = self._flash_font().render(self.flash_text, True, self.flash_color)
        pad_x, pad_y = 26, 14
        # The banner is composed on its own surface so set_alpha can fade the
        # panel, its border and the text together as one unit.
        banner = pygame.Surface((text.get_width() + pad_x * 2, text.get_height() + pad_y * 2), pygame.SRCALPHA)
        body = banner.get_rect()
        pygame.draw.rect(banner, (18, 16, 14, 225), body, border_radius=12)
        pygame.draw.rect(banner, (*self.flash_color, 210), body, width=2, border_radius=12)
        banner.blit(text, (pad_x, pad_y))
        banner.set_alpha(int(255 * fade))

        # Sits near the top of the board rather than dead centre, so it never
        # covers the squares a player is most likely to be looking at, and it
        # never blocks clicks -- it is drawn, not interactive.
        board = self.layout.board
        self.screen.blit(
            banner,
            banner.get_rect(center=(board.centerx, board.top + int(self.layout.square * 1.2))),
        )

    def _draw_history(self, x: int, y: int, bottom: int) -> None:
        """Move list, newest at the bottom, clipped to the space available.

        The capacity check is important: ``rows[-0:]`` would return the whole
        list rather than nothing, so the zero case is handled explicitly.
        """
        line_height = 19
        capacity = (bottom - y) // line_height
        if capacity <= 0:
            return
        history = self.game.history
        rows = [
            f"{index // 2 + 1:>3}. {history[index]:<9}" + (history[index + 1] if index + 1 < len(history) else "")
            for index in range(0, len(history), 2)
        ]
        for line in rows[-capacity:]:
            self.screen.blit(self.font_small.render(line, True, self.theme.muted), (x, y))
            y += line_height

    def _material_advantage(self) -> str:
        """e.g. ``'Alice +3'``, or empty when material is level.

        Compares the value of what each side has captured, which is equivalent
        to comparing what each side has left, and needs no board scan.
        """
        white_gain = sum(p.points for p in self.game.captured[WHITE])
        black_gain = sum(p.points for p in self.game.captured[BLACK])
        difference = white_gain - black_gain
        if difference == 0:
            return ""
        leader = self.game.players[WHITE] if difference > 0 else self.game.players[BLACK]
        return f"{leader} +{abs(difference)}"


class TextInput:
    """A single-line text field with click focus and a blinking caret.

    pygame has no widget toolkit, so this is written from scratch -- which for a
    two-field form is far less work than adding a GUI dependency.
    """

    MAX_LENGTH = 18  # Longer names would overflow the sidebar's player rows

    def __init__(self, label: str, text: str = "") -> None:
        self.label = label
        self.text = text
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.focused = False

    def handle_key(self, event: pygame.event.Event) -> None:
        """Apply one keypress to the field's text.

        ``event.unicode`` is used rather than mapping key codes, because it is
        already the character the keyboard layout produced -- so accented and
        non-US layouts work with no extra handling. The ``isprintable`` filter
        is what keeps control characters (Tab, Enter, Escape) out of the text.
        """
        if event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
        elif event.unicode and event.unicode.isprintable() and len(self.text) < self.MAX_LENGTH:
            self.text += event.unicode

    def draw(self, surface: pygame.Surface, theme: Theme, label_font, text_font) -> None:
        label = label_font.render(self.label, True, theme.muted)
        surface.blit(label, (self.rect.left, self.rect.top - label.get_height() - 6))
        # Focus is shown three ways at once (fill, border colour, border width)
        # so it is obvious which field typing will go to.
        pygame.draw.rect(surface, theme.button if self.focused else theme.panel, self.rect, border_radius=6)
        pygame.draw.rect(
            surface,
            theme.accent if self.focused else theme.muted,
            self.rect,
            width=2 if self.focused else 1,
            border_radius=6,
        )
        body = text_font.render(self.text, True, theme.text)
        surface.blit(body, (self.rect.left + 12, self.rect.centery - body.get_height() // 2))
        # Caret blink derived from the clock rather than a counter: it stays
        # steady regardless of frame rate, and needs no state.
        if self.focused and (pygame.time.get_ticks() // 500) % 2 == 0:
            caret_x = self.rect.left + 14 + body.get_width()
            pygame.draw.line(
                surface,
                theme.text,
                (caret_x, self.rect.top + 10),
                (caret_x, self.rect.bottom - 10),
                2,
            )


class NameEntryScreen:
    """Collects both player names before the game starts.

    Runs its own event loop on the *same* window surface the board will use, so
    switching between the two never recreates the window -- which is what lets a
    resize made here carry over into the game and back again.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        names: Tuple[str, str] = ("White", "Black"),
        theme: Optional[Theme] = None,
    ) -> None:
        self.screen = screen
        self.theme = theme or Theme()
        self.clock = pygame.time.Clock()
        # Prefilled with the current names so reopening this mid-session is an
        # edit rather than a retype.
        self.fields = [
            TextInput("WHITE PLAYER", names[0]),
            TextInput("BLACK PLAYER", names[1]),
        ]
        self.fields[0].focused = True
        # Rects are placeholders until _relayout sizes them to the window.
        self.start_button = Button("Start game", pygame.Rect(0, 0, 0, 0), "start")
        self.font_title = pygame.font.SysFont("segoeui,arial", 40, bold=True)
        self.font_label = pygame.font.SysFont("segoeui,arial", 15, bold=True)
        self.font_text = pygame.font.SysFont("segoeui,arial", 22)
        self.font_hint = pygame.font.SysFont("segoeui,arial", 16)
        self._relayout()

    def _relayout(self) -> None:
        """Centre the form for the current window size; re-run on every resize."""
        width, height = self.screen.get_size()
        field_width = min(380, width - 120)
        left = (width - field_width) // 2
        top = height // 2 - 60
        for index, field in enumerate(self.fields):
            field.rect = pygame.Rect(left, top + index * 96, field_width, 46)
        self.start_button.rect = pygame.Rect(left, top + 96 * len(self.fields) + 14, field_width, 46)

    def run(self) -> Optional[Tuple[str, str]]:
        """Returns the two names, or None if the window was closed.

        None is the "user quit" signal, which is why returning names and
        returning nothing are distinguishable -- the app loop uses that to decide
        whether to start a game or shut down.
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(
                        (max(MIN_WIDTH, event.w), max(MIN_HEIGHT, event.h)), pygame.RESIZABLE
                    )
                    self._relayout()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.start_button.rect.collidepoint(event.pos):
                        return self.names()
                    # Clicking anywhere sets focus: on a field if one was hit,
                    # otherwise clearing it. One line, no else branch needed.
                    for field in self.fields:
                        field.focused = field.rect.collidepoint(event.pos)
                elif event.type == pygame.KEYDOWN:
                    result = self._handle_key(event)
                    if result is not None:
                        return result
            self.draw()
            self.clock.tick(FPS)

    def _handle_key(self, event: pygame.event.Event) -> Optional[Tuple[str, str]]:
        """Returns the names when the form is submitted, else None.

        Enter in the first field advances to the second rather than submitting,
        which matches how forms normally behave and stops a player starting the
        game before naming Black.
        """
        if event.key == pygame.K_TAB or (event.key == pygame.K_RETURN and self.fields[0].focused):
            index = 0 if self.fields[1].focused else 1
            for position, field in enumerate(self.fields):
                field.focused = position == index
            return None
        if event.key == pygame.K_RETURN:
            return self.names()
        for field in self.fields:
            if field.focused:
                field.handle_key(event)
        return None

    def names(self) -> Tuple[str, str]:
        """Current field values, with defaults for anything left blank."""
        white = self.fields[0].text.strip() or "White"
        black = self.fields[1].text.strip() or "Black"
        return white, black

    def draw(self) -> None:
        theme = self.theme
        self.screen.fill(theme.background)
        width, _ = self.screen.get_size()

        title = self.font_title.render("Python Chess", True, theme.text)
        self.screen.blit(title, title.get_rect(centerx=width // 2, top=self.fields[0].rect.top - 150))
        subtitle = self.font_hint.render("Two players, one board. Who is playing?", True, theme.muted)
        self.screen.blit(subtitle, subtitle.get_rect(centerx=width // 2, top=self.fields[0].rect.top - 92))

        for field in self.fields:
            field.draw(self.screen, theme, self.font_label, self.font_text)
        self.start_button.draw(self.screen, self.font_text, theme, pygame.mouse.get_pos())

        hint = self.font_hint.render("Tab switches fields  -  Enter starts the game", True, theme.muted)
        self.screen.blit(hint, hint.get_rect(centerx=width // 2, top=self.start_button.rect.bottom + 18))
        pygame.display.flip()


def open_window() -> pygame.Surface:
    """Create the one resizable window both screens share."""
    pygame.init()
    pygame.display.set_caption("Python Chess")
    return pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT), pygame.RESIZABLE)


def run_app(names: Optional[Tuple[str, str]] = None) -> None:
    """Alternate between the name screen and the board until the user quits.

    Each screen owns its own event loop and returns when it is done, so this
    outer loop is the only place that knows both exist. That is what makes
    "Change players" mid-game work: the board loop exits with 'rename', control
    returns here, and the name screen reopens on the same window.

    ``screen`` is re-read from the GUI afterwards because a resize during play
    replaces the display surface, and the name screen must use the new one.
    """
    screen = open_window()
    current = names or ("White", "Black")
    while True:
        chosen = NameEntryScreen(screen, current).run()
        if chosen is None:  # window closed
            break
        current = chosen
        gui = ChessGUI(Game(*current), screen=screen)
        if gui.run() != "rename":
            break
        screen = gui.screen
        current = (gui.game.players[WHITE].name, gui.game.players[BLACK].name)
    pygame.quit()


def main() -> None:
    """GUI entry point. Optional positional args preseed the name fields."""
    given = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    names = (given[0], given[1]) if len(given) > 1 else None
    run_app(names)
