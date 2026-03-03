"""
Backward-compatible wrapper for the core pipeline module.

The implementation lives in `glint_pipeline/eval_gen.py`.
"""

from __future__ import annotations

# Re-export for existing imports (e.g. `import glint_pipeline_eval_gen as g`).
from glint_pipeline.eval_gen import *  # noqa: F401,F403


def main() -> int:
    from glint_pipeline.eval_gen import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())

