from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
CONFIG_DIR = SRC_DIR / "config"
FILES_DIR = ROOT_DIR / ".files"

# Ensure .files directory exists
FILES_DIR.mkdir(exist_ok=True)

def get_config_path(filename: str) -> Path:
    """Get the absolute path to a config file."""
    return CONFIG_DIR / filename

def get_file_path(filename: str) -> Path:
    """Get the absolute path to a generated file."""
    return FILES_DIR / filename 