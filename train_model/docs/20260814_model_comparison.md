# Calligraphy AI 模型測試比較

- 實驗日期：20260814
- 三個模型均使用固定 Kaggle test set；5-fold 則以五個 fold 的 best.pt 分別測試。
- Joint Accuracy 表示作者與書體必須在同一張圖片上同時預測正確。

## 整體比較

| 模型 | Epochs | Joint Accuracy | Combined Loss | 參數量 | 模型大小 | ms/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Separate Heads | 28 | 89.37% | 0.4201 | 17,182,426 | 65.57 MB | 0.584 |
| Shared Linear | 41 | 90.45% | 0.4014 | 8,792,282 | 33.56 MB | 0.309 |
| Average + Max Pooling | 50 | 70.02% | 1.1481 | 591,706 | 2.27 MB | 0.458 |

## 作者分類測試指標

| 模型 | Accuracy | Top-3 | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Loss | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Separate Heads | 92.27% | 98.38% | 0.8901 | 0.9029 | 0.8951 | 0.9236 | 0.2375 | 0.0061 |
| Shared Linear | 92.42% | 98.50% | 0.8957 | 0.9043 | 0.8993 | 0.9245 | 0.2353 | 0.0043 |
| Average + Max Pooling | 76.09% | 94.00% | 0.7109 | 0.7410 | 0.7136 | 0.7683 | 0.7643 | 0.1071 |

## 書體分類測試指標

| 模型 | Accuracy | Top-3 | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Loss | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Separate Heads | 93.48% | 99.58% | 0.8861 | 0.9258 | 0.9013 | 0.9368 | 0.1826 | 0.0074 |
| Shared Linear | 94.26% | 99.62% | 0.8986 | 0.9293 | 0.9117 | 0.9437 | 0.1660 | 0.0109 |
| Average + Max Pooling | 85.52% | 98.81% | 0.8049 | 0.8624 | 0.8142 | 0.8655 | 0.3838 | 0.0226 |

## Shared Linear 5-fold 測試結果

| Fold | Author Acc | Author Top-3 | Author Macro F1 | Style Acc | Style Top-3 | Style Macro F1 | Joint Acc | Combined Loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 92.86% | 98.53% | 0.9036 | 94.89% | 99.64% | 0.9220 | 91.17% | 0.3821 |
| 2 | 92.58% | 98.48% | 0.9028 | 94.63% | 99.64% | 0.9208 | 90.82% | 0.3925 |
| 3 | 92.94% | 98.55% | 0.9068 | 94.68% | 99.71% | 0.9180 | 91.15% | 0.3795 |
| 4 | 93.25% | 98.71% | 0.9099 | 95.05% | 99.64% | 0.9253 | 91.63% | 0.3718 |
| 5 | 93.12% | 98.56% | 0.9078 | 95.09% | 99.70% | 0.9237 | 91.53% | 0.3729 |

### 五折平均、標準差與 95% 信賴區間

| 指標 | Mean | Sample std | 95% CI |
| --- | ---: | ---: | ---: |
| Author Accuracy | 92.95% | 0.26% | [92.63%, 93.27%] |
| Author Top-3 | 98.56% | 0.09% | [98.46%, 98.67%] |
| Author Macro F1 | 0.9062 | 0.0030 | [0.9025, 0.9099] |
| Style Accuracy | 94.87% | 0.21% | [94.60%, 95.13%] |
| Style Top-3 | 99.67% | 0.04% | [99.62%, 99.71%] |
| Style Macro F1 | 0.9220 | 0.0028 | [0.9185, 0.9255] |
| Joint Accuracy | 91.26% | 0.33% | [90.86%, 91.66%] |
| Combined Loss | 0.3797 | 0.0083 | [0.3694, 0.3901] |

## 詳細檔案

- [Separate Heads test_metrics.json](../artifacts/runs/20260814_separate_heads_v1/test_metrics.json)
- [Shared Linear test_metrics.json](../artifacts/runs/20260814_shared_linear_v1/test_metrics.json)
- [Average + Max Pooling test_metrics.json](../artifacts/runs/20260814_avg_max_pool_v1/test_metrics.json)
