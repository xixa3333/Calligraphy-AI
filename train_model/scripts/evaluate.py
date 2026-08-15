import json

from calligraphy_ai.evaluation import evaluate_checkpoint
from calligraphy_ai.experiment import RUN_DIR


if __name__ == "__main__":
    metrics = evaluate_checkpoint(RUN_DIR)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
