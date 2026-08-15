import os

from calligraphy_ai.core.model import MODEL_ARCHITECTURE
from calligraphy_ai.paths import ARTIFACTS_DIR


RUN_DATE = os.environ.get("CALLIGRAPHY_RUN_DATE", "20260814")
RUN_VERSION = os.environ.get("CALLIGRAPHY_RUN_VERSION", "v1")
RUN_NAME = f"{RUN_DATE}_{MODEL_ARCHITECTURE}_{RUN_VERSION}"
RUN_DIR = ARTIFACTS_DIR / "runs" / RUN_NAME
BEST_MODEL_PATH = RUN_DIR / "best.pt"
LAST_CHECKPOINT_PATH = RUN_DIR / "last.pt"
TRAINING_PLOT_PATH = RUN_DIR / "training_result.png"
