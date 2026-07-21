from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
LOGS_DIR = ARTIFACTS_DIR / "logs"
WEIGHTS_DIR = ARTIFACTS_DIR / "weights"
