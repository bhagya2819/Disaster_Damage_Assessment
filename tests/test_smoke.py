"""Phase-0 smoke test: the package imports and paths resolve."""

from __future__ import annotations

from pathlib import Path


def test_package_imports() -> None:
    import src  # noqa: F401

    assert hasattr(src, "__version__")


def test_subpackages_import() -> None:
    from src import (  # noqa: F401
        analysis,
        data,
        dip,
        eval,
        inference,
        models,
        pipelines,
        preprocess,
        train,
        utils,
    )


def test_paths_resolve() -> None:
    from src.utils.paths import DATA_DIR, PROJECT_ROOT

    assert isinstance(PROJECT_ROOT, Path)
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "PRD.md").is_file(), "PRD.md should sit at the project root"
    assert isinstance(DATA_DIR, Path)


def test_logger_works() -> None:
    from src.utils.logging import get_logger

    log = get_logger("dda.test")
    log.info("smoke test logger wired up")
