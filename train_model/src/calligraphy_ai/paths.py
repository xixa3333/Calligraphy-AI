from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
PREPROCESSED_DIR = ARTIFACTS_DIR / "preprocessed"
RUNS_DIR = ARTIFACTS_DIR / "runs"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

# Metadata alias used by the training and evaluation scripts.
LOGS_DIR = METADATA_DIR
