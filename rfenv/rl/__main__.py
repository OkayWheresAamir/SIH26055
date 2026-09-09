"""`python -m rfenv.rl ...` -- the actual CLI lives in `_cli.py`.

Kept separate from `_cli.py` deliberately: `__init__.py` re-exports `main` for
`rfenv.rl.main(...)` to work, and importing `__main__` from `__init__.py`
triggers Python's own "found in sys.modules before execution" warning under
`python -m`. A thin shim avoids it.
"""

from __future__ import annotations

import sys

from rfenv.rl._cli import main

if __name__ == "__main__":
    sys.exit(main())
