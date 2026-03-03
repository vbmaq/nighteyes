"""Backward-compatible UI entry point.

The implementation lives in `apps/templatemaker_manual_app.py`.
"""

from __future__ import annotations


def main() -> None:
    from apps.templatemaker_manual_app import main as _main

    return _main()


if __name__ == "__main__":
    main()

