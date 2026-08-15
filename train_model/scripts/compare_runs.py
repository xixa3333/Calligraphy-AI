import json

import torch

from calligraphy_ai.paths import ARTIFACTS_DIR

RUN_DATE = "20260814"
RUN_VERSION = "v1"
RUNS_ROOT = ARTIFACTS_DIR / "runs"
DOCS_DIR = ARTIFACTS_DIR.parent / "docs"
OUTPUT_PATH = DOCS_DIR / f"{RUN_DATE}_model_comparison.md"
ARCHITECTURES = ("separate_heads", "shared_linear", "avg_max_pool")
DISPLAY_NAMES = {
    "separate_heads": "Separate Heads",
    "shared_linear": "Shared Linear",
    "avg_max_pool": "Average + Max Pooling",
}


def percent(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def number(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def load_runs():
    runs = []
    for architecture in ARCHITECTURES:
        run_dir = RUNS_ROOT / f"{RUN_DATE}_{architecture}_{RUN_VERSION}"
        metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        runs.append(
            {
                "architecture": architecture,
                "name": DISPLAY_NAMES[architecture],
                "dir": run_dir,
                "metrics": metrics,
                "epochs": checkpoint["epoch"] + 1,
            }
        )
    return runs


def task_table(lines, runs, task):
    title = "作者分類" if task == "author" else "書體分類"
    lines.extend(
        [
            f"## {title}測試指標",
            "",
            "| 模型 | Accuracy | Top-3 | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Loss | ECE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in runs:
        metric = run["metrics"][task]
        lines.append(
            f"| {run['name']} | {percent(metric['accuracy'])} | "
            f"{percent(metric['top3_accuracy'])} | {number(metric['macro_precision'])} | "
            f"{number(metric['macro_recall'])} | {number(metric['macro_f1'])} | "
            f"{number(metric['weighted_f1'])} | {number(metric['loss'])} | "
            f"{number(metric['ece_15_bins'])} |"
        )
    lines.append("")


def add_five_fold(lines):
    run_dir = RUNS_ROOT / f"{RUN_DATE}_shared_linear_5fold_{RUN_VERSION}"
    summary_path = run_dir / "cross_validation_metrics.json"
    lines.extend(["## Shared Linear 5-fold 測試結果", ""])
    if not summary_path.exists():
        lines.extend(["5-fold 尚未全部完成；完成後重新執行 `python scripts/compare_runs.py` 即會補上。", ""])
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lines.extend(
        [
            "| Fold | Author Acc | Author Top-3 | Author Macro F1 | Style Acc | Style Top-3 | Style Macro F1 | Joint Acc | Combined Loss |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for index, metric in enumerate(summary["fold_test_metrics"], start=1):
        lines.append(
            f"| {index} | {percent(metric['author']['accuracy'])} | "
            f"{percent(metric['author']['top3_accuracy'])} | {number(metric['author']['macro_f1'])} | "
            f"{percent(metric['style']['accuracy'])} | {percent(metric['style']['top3_accuracy'])} | "
            f"{number(metric['style']['macro_f1'])} | {percent(metric['joint']['accuracy'])} | "
            f"{number(metric['combined_loss'])} |"
        )

    aggregate = summary["test_aggregate"]
    lines.extend(
        [
            "",
            "### 五折平均、標準差與 95% 信賴區間",
            "",
            "| 指標 | Mean | Sample std | 95% CI |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    important = (
        ("author.accuracy", "Author Accuracy", True),
        ("author.top3_accuracy", "Author Top-3", True),
        ("author.macro_f1", "Author Macro F1", False),
        ("style.accuracy", "Style Accuracy", True),
        ("style.top3_accuracy", "Style Top-3", True),
        ("style.macro_f1", "Style Macro F1", False),
        ("joint.accuracy", "Joint Accuracy", True),
        ("combined_loss", "Combined Loss", False),
    )
    for key, label, as_percent in important:
        item = aggregate[key]
        formatter = percent if as_percent else number
        lines.append(
            f"| {label} | {formatter(item['mean'])} | {formatter(item['sample_std'])} | "
            f"[{formatter(item['95_ci_lower'])}, {formatter(item['95_ci_upper'])}] |"
        )
    lines.append("")


def main():
    runs = load_runs()
    lines = [
        "# Calligraphy AI 模型測試比較",
        "",
        f"- 實驗日期：{RUN_DATE}",
        "- 三個模型均使用固定 Kaggle test set；5-fold 則以五個 fold 的 best.pt 分別測試。",
        "- Joint Accuracy 表示作者與書體必須在同一張圖片上同時預測正確。",
        "",
        "## 整體比較",
        "",
        "| 模型 | Epochs | Joint Accuracy | Combined Loss | 參數量 | 模型大小 | ms/image |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        metrics = run["metrics"]
        joint = metrics.get("joint", {}).get("accuracy")
        lines.append(
            f"| {run['name']} | {run['epochs']} | {percent(joint)} | "
            f"{number(metrics['combined_loss'])} | {metrics['parameter_count']:,} | "
            f"{metrics['model_size_bytes'] / 1024**2:.2f} MB | "
            f"{metrics['inference_ms_per_image']:.3f} |"
        )
    lines.append("")
    task_table(lines, runs, "author")
    task_table(lines, runs, "style")
    add_five_fold(lines)
    lines.extend(["## 詳細檔案", ""])
    for run in runs:
        relative = f"../artifacts/runs/{run['dir'].name}"
        lines.append(f"- [{run['name']} test_metrics.json]({relative}/test_metrics.json)")
    lines.append("")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
