"""Backward-compatible UI entry point.

The implementation lives in `apps/glint_pipeline_preview_ui_app.py`.
"""

from __future__ import annotations


def main() -> None:
    from apps.glint_pipeline_preview_ui_app import main as _main

    return _main()


if __name__ == "__main__":
    main()

