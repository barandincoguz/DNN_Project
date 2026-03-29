# Report 02: Huber Loss
**Date:** 2026-03-23
**Baseline MAE:** 0.7951 (Report 01)

## Change
MSE → Huber Loss (delta=1.0). Outlier'lara karşı daha dayanıklı olması beklendi.

## Config
- Loss: Huber (delta=1.0) — diğer tüm parametreler baseline ile aynı

## Results
| Fold | Val MAE | Early Stop Epoch |
|------|---------|-----------------|
| 1    | 0.9282  | 117             |
| 2    | 0.7106  | 148             |
| 3    | 0.6877  | 129             |
| 4    | 0.6945  | 188             |
| 5    | 0.9141  | 106             |
| **Mean** | **0.7870 ± 0.1099** | |

## Analysis
- Mean MAE hafif iyileşti (0.7951 → 0.7870) ama fold varyansı arttı (0.069 → 0.110).
- Fold 3-4 baseline'dan belirgin iyi (0.68-0.69), Fold 1-5 kötüleşti (0.91-0.93).
- Huber Loss gradient'leri küçülttüğü için bazı fold'larda yeterince öğrenememiş olabilir.
- Kararsız iyileşme — şimdilik MSE'ye geri dönüp diğer iyileştirmelere odaklanalım.

## Next Steps
- MSE'ye geri dön (daha stabil)
- KNN Imputer overflow sorununu çöz (f4 ölçek farkı)
- Patience artır veya architecture değiştir
