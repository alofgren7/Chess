"""Entry point.

    python main.py             launch the graphical board
    python main.py --console   play in the terminal instead
"""

import sys


def _run_console() -> None:
    # Windows consoles often default to a codepage that cannot encode the chess
    # glyphs. Switching stdout to UTF-8 lets them print; the board renderer has
    # its own letter fallback for terminals where this fails.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass
    from chess.game import main

    main()


def _run_gui() -> None:
    # Imported lazily so a missing pygame is a graceful downgrade to the text
    # version rather than an import error at startup.
    try:
        from chess.gui import main
    except ImportError:
        print("pygame is required for the GUI.  Install it with:  pip install pygame")
        print("Falling back to the console version.\n")
        _run_console()
        return
    main()


if __name__ == "__main__":
    if "--console" in sys.argv or "-c" in sys.argv:
        _run_console()
    else:
        _run_gui()
