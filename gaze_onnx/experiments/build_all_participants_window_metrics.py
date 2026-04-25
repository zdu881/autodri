from __future__ import annotations

import sys
from pathlib import Path


_HERE = Path(__file__).resolve()
for _parent in (_HERE.parent, *_HERE.parents):
    if (_parent / "src" / "autodri").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from autodri.workflows.build_all_participants_window_metrics import *  # noqa: F401,F403


if __name__ == "__main__":
    main()

