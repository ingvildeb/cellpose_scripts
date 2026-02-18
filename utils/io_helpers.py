from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        # Handle the error or exit
        import sys
        sys.exit('Error: This program requires either tomllib or tomli but neither is available')


# -------------------------
# PATH NORMALIZATION
# -------------------------
def normalize_user_path(p: str | Path) -> Path:
    """Normalize a user-provided path string into a Path object."""
    if isinstance(p, Path):
        return p
    return Path(p.replace("\\", "/")).expanduser()


# -------------------------
# VALIDATION HELPERS
# -------------------------
def require_dir(path: str | Path, name: str = "Directory") -> Path:
    """Ensure that a directory exists and is a folder."""
    p = normalize_user_path(path)
    if not p.exists():
        raise RuntimeError(f"{name} does not exist:\n{p}")
    if not p.is_dir():
        raise RuntimeError(f"{name} is not a directory:\n{p}")
    return p


def require_file(path: str | Path, name: str = "File") -> Path:
    """Ensure that a file exists and is a regular file."""
    p = normalize_user_path(path)
    if not p.exists():
        raise RuntimeError(f"{name} does not exist:\n{p}")
    if not p.is_file():
        raise RuntimeError(f"{name} is not a file:\n{p}")
    return p


def require_subpath(parent: Path, sub: str, name: str) -> Path:
    """Ensure that a required subpath exists inside a parent directory."""
    p = parent / sub
    if not p.exists():
        raise RuntimeError(f"Missing {name} in:\n{parent}\nExpected:\n{p}")
    return p


# -------------------------
# CONFIG LOADER
# -------------------------
def load_script_config(script_path: Path, config_basename: str) -> dict[str, Any]:
    """
    Load config from configs/<basename>_local.toml, else _template.toml.
    """
    config_dir = script_path.parent / "configs"
    local_path = config_dir / f"{config_basename}_local.toml"
    template_path = config_dir / f"{config_basename}_template.toml"

    config_path = local_path if local_path.exists() else template_path
    if not config_path.exists():
        raise FileNotFoundError(
            "No config file found.\n"
            f"Expected:\n{local_path}\nOR\n{template_path}"
        )

    with config_path.open("rb") as f:
        cfg: dict[str, Any] = tomllib.load(f)

    print(f"Using config: {config_path.name}")
    return cfg
