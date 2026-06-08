"""Entry point for `python -m gator`."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

# Import application code after load_dotenv() so the environment is populated first.
from gator.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
