# Shared Linear 5-fold 穩定性評估

## 評估結論

Shared Linear 模型的五折結果整體穩定，可視為對資料切分不敏感、泛化表現一致的模型。作者、書體與聯合準確率的跨折樣本標準差都低於 0.33 個百分點；Macro F1 的標準差也低於 0.30 個百分點。五折之間沒有出現單一 fold 明顯崩潰或嚴重偏離平均的情況。

正式部署可採用 validation loss 最低的 Fold 5。Fold 5 同時具有最高的 validation 作者／書體準確率，固定 test set 表現也位於五折前段，選擇依據合理。若更重視 test set 的單次最佳結果，Fold 4 略優；但不應利用 test set 反向挑選正式模型，因此仍建議依 validation 指標選 Fold 5。

## 實驗設定

- 模型架構：Shared Linear 多任務 CNN
- 交叉驗證：5-fold stratified cross-validation
- 分折資料：Kaggle 原始 train set
- 最終測試：每個 fold 的 `best.pt` 均使用同一份 Kaggle test set（21,007 張）
- 模型選擇：各 fold 以最低 validation loss 保存 `best.pt`
- 統計方式：五折平均、樣本標準差（ddof=1）、95% Student's t 信賴區間
- 隨機種子：42

資料增強僅套用於每一折的訓練子集，validation 與固定 test set 均不套用資料增強，因此沒有 augmentation 洩漏到評估資料。

## 各折 Validation 結果

| Fold | Best Epoch | Val Loss | Author Accuracy | Style Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 49 | 0.4013 | 92.59% | 94.76% |
| 2 | 43 | 0.3918 | 92.59% | 94.75% |
| 3 | 40 | 0.4285 | 92.32% | 94.37% |
| 4 | 43 | 0.3872 | 92.74% | 94.94% |
| 5 | 49 | **0.3809** | **92.84%** | **95.22%** |
| 平均 | 44.8 | 0.3979 | 92.62% | 94.81% |
| 樣本標準差 | 4.02 | 0.0186 | 0.20 pp | 0.31 pp |

Validation 指標的跨折範圍很小：作者準確率相差 0.52 個百分點，書體準確率相差 0.85 個百分點。Fold 3 相對較弱，但差距沒有大到構成不穩定或分折失敗。

Best epoch 分布在 40～49，平均為 44.8。這表示不同資料切分的收斂時間略有差異，但都在設定的 50 epochs 上限內進入最佳區域。最大 50 epochs 的設定合理，且 `best.pt` 避免採用最佳點之後的權重。

## 各折固定 Test Set 結果

| Fold | Author Acc | Author Top-3 | Author Macro F1 | Style Acc | Style Top-3 | Style Macro F1 | Joint Acc | Combined Loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 92.86% | 98.53% | 0.9036 | 94.89% | 99.64% | 0.9220 | 91.17% | 0.3821 |
| 2 | 92.58% | 98.48% | 0.9028 | 94.63% | 99.64% | 0.9208 | 90.82% | 0.3925 |
| 3 | 92.94% | 98.55% | 0.9068 | 94.68% | 99.71% | 0.9180 | 91.15% | 0.3795 |
| 4 | **93.25%** | **98.71%** | **0.9099** | 95.05% | 99.64% | **0.9253** | **91.63%** | **0.3718** |
| 5 | 93.12% | 98.56% | 0.9078 | **95.09%** | **99.70%** | 0.9237 | 91.53% | 0.3729 |

## 五折平均與不確定性

`pp` 代表百分點。Accuracy 的 mean、std 與 CI 以下均以百分比表示。

| 指標 | Mean | Sample std | 95% CI | 跨折範圍 |
| --- | ---: | ---: | ---: | ---: |
| Author Accuracy | 92.95% | 0.26 pp | [92.63%, 93.27%] | 0.67 pp |
| Author Top-3 | 98.56% | 0.09 pp | [98.46%, 98.67%] | 0.23 pp |
| Author Macro F1 | 0.9062 | 0.0030 | [0.9025, 0.9099] | 0.0072 |
| Style Accuracy | 94.87% | 0.21 pp | [94.60%, 95.13%] | 0.47 pp |
| Style Top-3 | 99.67% | 0.04 pp | [99.62%, 99.71%] | 0.07 pp |
| Style Macro F1 | 0.9220 | 0.0028 | [0.9185, 0.9255] | 0.0073 |
| Joint Accuracy | 91.26% | 0.33 pp | [90.86%, 91.66%] | 0.81 pp |
| Combined Loss | 0.3797 | 0.0083 | [0.3694, 0.3901] | 0.0207 |

### 穩定性判讀

1. **作者分類穩定**：Accuracy 標準差僅 0.26 pp，Macro F1 標準差為 0.0030。類別不平衡下的表現並未隨 fold 大幅改變。
2. **書體分類更穩定**：Accuracy 標準差為 0.21 pp，Top-3 幾乎固定在 99.6%～99.7%。
3. **多任務聯合結果穩定**：Joint Accuracy 平均 91.26%，標準差 0.33 pp。雖然它比單一任務更嚴格，仍未出現明顯跨折波動。
4. **Loss 波動可接受**：Combined loss 的標準差為 0.0083。Fold 2 loss 稍高，但 Accuracy 與 F1 沒有同步惡化，模型並未失效。
5. **校準整體良好**：Author ECE 平均 0.0080，Style ECE 平均 0.0116，表示預測信心與實際正確率大致一致。Fold 4、5 的信心較高，ECE 也略高，但仍在低值範圍。

## 與一般 Shared Linear 訓練比較

| 指標 | 一般訓練 | 5-fold 平均 | 差異 |
| --- | ---: | ---: | ---: |
| Author Accuracy | 92.42% | 92.95% | +0.53 pp |
| Author Macro F1 | 0.8993 | 0.9062 | +0.0069 |
| Style Accuracy | 94.26% | 94.87% | +0.61 pp |
| Style Macro F1 | 0.9117 | 0.9220 | +0.0103 |
| Joint Accuracy | 90.45% | 91.26% | +0.81 pp |
| Combined Loss | 0.4014 | 0.3797 | -0.0217 |

五折平均全面略高於原本單次切分模型，且提升幅度大於跨折標準差，顯示原本的單次 train/validation 切分可能略偏保守；Shared Linear 的實際泛化能力更接近五折平均結果。

## 模型選擇建議

- **正式模型建議：Fold 5 `best.pt`**。它由最低 validation loss（0.3809）選出，沒有利用 test set 做模型選擇，因此方法上較嚴謹。
- Fold 4 在固定 test set 的 Combined loss、Author Accuracy、Author Macro F1、Style Macro F1 與 Joint Accuracy略高，但這些是測試後觀察，不建議據此取代 validation 選出的 Fold 5。
- 如果未來追求更高穩定度，可以對五個 fold 的 logits 做 ensemble；代價是約五倍推論成本與模型容量，不適合目前的輕量網頁部署需求。

## 統計解讀限制

五個模型都使用同一份固定 Kaggle test set，而且五個訓練 fold 彼此共享部分訓練樣本，因此五組 test 指標並非完全獨立。報告中的 sample std 與 95% t CI 最適合解讀為「模型對訓練／validation 切分的敏感度」，不應視為整個未知母體的嚴格獨立抽樣信賴區間。

若要估計 test set 抽樣不確定性，可以另外對固定 test set 做 paired bootstrap；這不影響目前對五折訓練穩定性的結論。

## 最終判定

**穩定性：良好。** Shared Linear 在五個資料切分下均維持約 92.6%～93.2% 的作者測試準確率、94.6%～95.1% 的書體測試準確率，以及 90.8%～91.6% 的 Joint Accuracy。跨折差異小、Macro F1 一致、校準誤差低，沒有明顯過度依賴特定資料切分的跡象，適合作為正式部署模型。

## 來源檔案

- [五折完整彙整](../artifacts/runs/20260814_shared_linear_5fold_v1/cross_validation_metrics.json)
- [三模型比較](20260814_model_comparison.md)
- 各 fold 的 `test_metrics.json`、classification reports、confusion matrices 與 training curves 位於 `../artifacts/runs/20260814_shared_linear_5fold_v1/fold_1` 至 `fold_5`。
