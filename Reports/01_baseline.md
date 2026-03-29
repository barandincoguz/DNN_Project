# Report 01: Baseline
**Date:** 2026-03-23
**Baseline MAE:** N/A (first run)

## Config
- Architecture: [128, 256, 128, 64] (76,033 params)
- Activation: SiLU, BatchNorm: True, Dropout: 0.15
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingWarmRestarts (T_0=50)
- Loss: MSE, Patience: 25, Max Epochs: 500
- Data: f5 dropped, KNN Impute (k=10), log1p(f3), StandardScaler

## Results
| Fold | Val MAE | Early Stop Epoch |
|------|---------|-----------------|
| 1    | 0.8690  | 152             |
| 2    | 0.7245  | 165             |
| 3    | 0.7035  | 150             |
| 4    | 0.8159  | 121             |
| 5    | 0.8625  | 154             |
| **Mean** | **0.7951 ± 0.0690** | |

## Analysis
- Fold varyansı yüksek (0.0690) — Fold 3 en iyi (0.70), Fold 1 en kötü (0.87).
- Early stopping 121-165 arası — model hala öğrenebilir, patience artırılabilir.
- KNN Imputer overflow uyarıları var — f4 ölçeği (0-978) diğer feature'lardan çok farklı.

## Next Steps
- KNN Imputer uyarısını çöz (imputation öncesi ölçekleme veya SimpleImputer dene)
- Huber Loss dene (outlier'lara karşı dayanıklılık)
- Learning rate ve architecture tuning
