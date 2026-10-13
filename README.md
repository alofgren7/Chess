# Python Chess

A complete two-player chess game written in Python with object-oriented design.
Every rule of standard chess is implemented and verified, including castling, en
passant, promotion, and all draw conditions.

Play in a resizable graphical window with the mouse, or in the terminal.

```
python main.py             # graphical board (default)
python main.py --console   # text version
```

---

## Contents

- [Features](#features)
- [Requirements and setup](#requirements-and-setup)
- [Playing the game](#playing-the-game)
- [Project structure](#project-structure)
- [How it works](#how-it-works)
- [Correctness and testing](#correctness-and-testing)
- [Using the engine as a library](#using-the-engine-as-a-library)
- [Extending the project](#extending-the-project)

---

## Features

### Complete rules

| Rule | Status |
| --- | --- |
| Piece movement, captures, blocking | Yes |
| Check, checkmate, stalemate | Yes |
| Castling (both sides, all five legality conditions) | Yes |
| En passant (including the one-move timing window) | Yes |
| Promotion to queen, rook, bishop or knight | Yes |
| Absolute pins and discovered checks | Yes |
| Fifty-move rule and threefold repetition (claimable) | Yes |
| Seventy-five move rule and fivefold repetition (automatic) | Yes |
| Insufficient material (dead position) | Yes |
| Resignation and draw by agreement | Yes |

### Graphical interface

- Top-down board, **click to select, click to move**, with legal destinations
  shown as dots and capturable pieces ringed
- **Fully resizable window** — the board, pieces, labels, and panel all scale
- **Turn announcement** that fades in over the board after every move, so two
  players sharing one screen always know whose turn it is
- **Promotion picker** that drops down at the promotion square
- Highlights for the last move, the selected piece, and a king in check
- Sidebar with both players, captured pieces, material advantage, live move list
  in standard algebraic notation, and game status
- Board flip, plus auto-flip to rotate the board to whoever is on move
- Unlimited undo
- Player name entry at startup, changeable mid-game

### Console interface

A full text version with the same rules: coordinate move entry, a legal-move
lister, unicode or ASCII pieces, board flipping, and the same undo/draw/resign
commands.

---

## Requirements and setup

- **Python 3.8 or newer**
- **pygame** (only for the graphical version)

```powershell
pip install pygame
python main.py
```

The console version has no third-party dependencies. If pygame is not installed,
`python main.py` prints an explanation and falls back to the terminal game
automatically.

Optionally, preseed the player names:

```powershell
python main.py Alice Bob
```

---

## Playing the game

### Graphical controls

| Input | Action |
| --- | --- |
| Left click a piece | Select it and show its legal moves |
| Left click a destination | Play the move |
| Left click another of your pieces | Reselect |
| Right click / `Esc` | Deselect |
| `U` | Undo the last move |
| `F` | Flip the board |
| `A` | Toggle auto-flip (board rotates to the player on move) |
| `N` | New game with the same players |
| `P` | Change player names |
| `Q` | Quit |

Castling is played by clicking the king and then its destination two squares
away. Promotion opens a picker; clicking outside it cancels the move rather than
guessing a piece for you.

**Resign** and **Offer draw** require a second click to confirm. When the
fifty-move or threefold repetition rule applies, the draw button changes to
name the rule being claimed.

### Console commands

| Command | Action |
| --- | --- |
| `e2e4` or `e2 e4` | Make a move |
| `e7e8q` | Promote (`q`, `r`, `b`, `n`) |
| `O-O` / `O-O-O` | Castle |
| `moves [square]` | List legal moves, optionally for one square |
| `board` / `history` | Redraw the board / show the move list |
| `undo` | Take back the last move |
| `flip` / `unicode` | Board orientation / piece style |
| `draw` / `resign` | Offer or claim a draw / resign |
| `help` | Show all commands |
| `quit` | Exit |

---

## Project structure

```
python_chess/
├── main.py              Entry point; chooses GUI or console
└── chess/
    ├── __init__.py      Package exports
    ├── move.py          Squares, coordinates, the Move object
    ├── pieces.py        Piece hierarchy and per-piece move generation
    ├── board.py         Position, make/undo, legality, notation, FEN
    ├── game.py          Players, history, results, and the console UI
    └── gui.py           The pygame front-end
```

Each layer depends only on the ones above it in that list. Nothing in `board.py`
knows a user interface exists, and nothing in `gui.py` knows a chess rule.

---

## How it works

Three design decisions shape the whole codebase.

### 1. Make, test, unmake

Determining whether a move is legal is the hard part of a chess engine. Rather
than analytically computing pins, discovered attacks and check evasions, the
board simply **plays each candidate move, asks whether the mover's king is now
attacked, and takes it back**:

```python
for move in self.pseudo_legal_moves(color):
    self.make_move(move)
    if not self.in_check(color):
        legal.append(move)
    self.undo_move()
```

This is slower than a purpose-built legal move generator, but it is far simpler
and correct by construction for every awkward case at once — absolute pins,
discovered checks, kings stepping along an attacker's ray, and the notorious
position where an en passant capture exposes its own king along a rank. None of
those need a single line of dedicated code.

The price is that `undo_move` must restore state **perfectly**, including things
that are easy to overlook: the captured piece, the en passant square, the
halfmove clock, and the `has_moved` flags. A `_UndoState` snapshot is pushed
before every mutation to guarantee that. This matters more than it appears — a
missed field there does not produce a visible error, it silently corrupts move
generation, because legality testing undoes thousands of moves the player never
sees.

### 2. Attacks and moves are different questions

Each piece answers two questions:

- `attacked_squares()` — which squares does this piece bear down on? Used for
  check detection and for castling's "may not pass through an attacked square"
  rule.
- `generate_moves()` — which moves could it actually play?

For every piece except the pawn these are nearly identical, so the base class
implements the second in terms of the first. **The pawn is the exception that
justifies the split**: it moves straight forward but captures diagonally.
Conflating the two would let pawns give check down a file.

Piece movement itself reduces to two shared patterns — sliding (bishop, rook,
queen) and stepping (knight, king) — so the traversal logic is written once and
each piece supplies only its direction offsets.

Castling lives on the `King` rather than on the board, because the king is what
the player actually moves. Its five conditions are checked explicitly rather
than left to the generic legality filter, since that filter only inspects the
*final* position and would happily allow a king to castle out of, or straight
through, check.

### 3. One layout, derived from the window

In the GUI, no pixel coordinate is written down twice. A `Layout` object is
computed from the current window size and provides the square size, the board
rectangle, and the sidebar rectangle. Every draw call and every hit test reads
from it.

That single indirection is what makes resizing a small feature rather than a
rewrite: handling `VIDEORESIZE` means building a new `Layout` and rebuilding the
buttons. It also keeps the two coordinate mappings — `square_at` (pixel to
square) and `square_rect` (square to pixel) — as exact inverses, which is where
board-flipping bugs otherwise hide.

The GUI **never decides whether a move is legal**. It asks the board for the
legal move list once per position, and clicking simply filters that list. Board
flipping is purely a rendering concern; the underlying grid is never rearranged.

### Other notes

- **Coordinates.** A square is `(row, col)` with row 0 = rank 1 and col 0 =
  file a, so White's pawns move in the `+row` direction. Many programs put rank
  8 at row 0 to match FEN's reading order, but then every pawn calculation is
  inverted relative to how players think about the board.
- **Castling rights** are not stored as board flags. They are derived from the
  `has_moved` attribute of the kings and rooks themselves, so there is exactly
  one copy of that truth and it cannot drift out of sync with the pieces.
- **Pieces do not know where they are.** Position is the board's
  responsibility, and a piece is passed its square when it needs one. That
  removes an entire class of synchronisation bug from make/undo.
- **Pieces are drawn from font glyphs**, not image files, so the project ships
  no assets. The renderer detects a font that actually contains the chess
  characters (using `Font.metrics()`, since a missing glyph still renders as a
  box and still reports a width) and falls back to drawing letters if none is
  available.

---

## Correctness and testing

Chess rules are easy to get *almost* right. The standard way to prove a move
generator correct is **perft**: count all legal move sequences to a given depth
and compare against published values.

This implementation matches the known counts for the starting position and for
the standard test positions that specifically target castling, en passant,
promotion and pin edge cases:

| Position | Depth 1 | Depth 2 | Depth 3 | Depth 4 |
| --- | --- | --- | --- | --- |
| Starting position | 20 | 400 | 8,902 | 197,281 |
| Kiwipete | 48 | 2,039 | 97,862 | — |
| Endgame (rook/pawn) | 14 | 191 | 2,812 | 43,238 |
| Promotion-heavy | 6 | 264 | 9,467 | — |
| Mirrored/tactical | 44 | 1,486 | 62,379 | — |

To reproduce:

```python
from chess.board import Board

def perft(board, depth):
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves():
        board.make_move(move)
        total += perft(board, depth - 1)
        board.undo_move()
    return total

board = Board()
for depth in range(1, 5):
    print(depth, perft(board, depth))
```

`Board.from_fen()` exists primarily to support this. It lets any position — a
test case or a bug report — be set up from a one-line string instead of by
playing a sequence of moves to reach it.

If your own numbers ever disagree, run a **perft divide**: compare the per-move
subtotals against a known-good engine to binary-search down to the exact
position that breaks.

---

## Using the engine as a library

The engine is usable on its own, with no UI:

```python
from chess import Board, Game, parse_square

game = Game("Alice", "Bob")

move = game.board.find_move(parse_square("e2"), parse_square("e4"))
print(game.play_move(move))          # 'e4'

for move in game.board.legal_moves():
    print(game.board.san(move), move.uci)

print(game.board)                    # ASCII diagram
print(game.board.in_check("black"))
print(game.formatted_history())
```

Useful entry points:

| Call | Purpose |
| --- | --- |
| `Board()` / `Board.from_fen(fen)` | Create a position |
| `board.legal_moves(color=None)` | All legal moves for the side to move |
| `board.find_move(start, end, promotion=None)` | Look up a move by square |
| `board.make_move(m)` / `board.undo_move()` | Play and rewind |
| `board.san(m)` | Standard algebraic notation for a move |
| `board.in_check(color)` / `is_checkmate()` / `is_stalemate()` | Position status |
| `game.play_move(m)` | Play, record history, update the result |
| `game.undo()` | Take back one ply, including bookkeeping |
| `game.claimable_draw()` | Name of a draw the player could claim, if any |

Note that `legal_moves()` returns a list rather than an iterator on purpose: the
board is mutated during generation, so a lazy iterator would interleave
make/undo calls with the caller's own work and corrupt the position.

---

## Extending the project

The layering makes several additions straightforward:

- **A computer opponent.** Everything needed is already here — `legal_moves()`,
  `make_move`/`undo_move`, and `Piece.points` for material. A minimax search
  with alpha-beta pruning slots in above `Board` without touching it. Expect to
  need a faster attack map first; `is_attacked` is deliberately brute-force.
- **Chess clocks.** `Player` exists partly as a home for this.
- **Save and load games.** `Board.from_fen()` already handles import; PGN export
  is mostly done, since `game.history` is a list of SAN strings.
- **Drag-and-drop pieces.** Add `MOUSEMOTION` and `MOUSEBUTTONUP` handling in
  `ChessGUI`; the existing `_apply_move` funnel means nothing else changes.
- **Themes.** Every colour is already in the `Theme` dataclass.
- **Move sounds, animation, arrows.** All purely `gui.py` concerns.

The source is thoroughly commented throughout, with particular attention to
explaining *why* a given approach was chosen rather than just what the code does.
