# CLAUDE.md - DNN Universal Function Approximation Project

## Temel Felsefe
- **KISS (Keep It Simple, Stupid)** - Her zaman isii halledecek EN BASiT kodu yaz. Profesör her satiri soruyor, açıklayamayacağın kod yazma.
- **Over-engineering YASAK** - Gereksiz abstraction, utility, helper yok. 3 satırla olan işi class'a sarma.
- **Detaylı EDA her zaman** - Veri değiştiğinde veya yeni deney öncesi mutlaka EDA yap.

## Yarışma Kuralları
- **Görev:** Universal Approximation Theorem - DNN ile profesörün arka planda ürettiği kompleks fonksiyonu öğrenmek
- **Train:** 1600 satır (id, f1, f2, f3, f4, f5, target)
- **Test:** 400 satır (id, f1, f2, f3, f4, f5) - profesörün kasasında ~1000 test verisi var
- **Kazanma kriteri:** Profesörün gizli test verisinde en düşük hata
- **Platform:** MacBook M4 Pro (MPS backend)
- **Framework:** PyTorch

## Kritik Kurallar
- **f5 TUZAK feature** - ASLA kullanma. Sadece [f1, f2, f3, f4] kullan.
- **Ensemble YASAK** - Tek DNN mimarisi. 5-fold CV ortalaması ensemble sayılmaz.
- **Per-fold preprocessing** - Imputer ve scaler her fold'da ayrı fit et (data leakage önle).
- **Ana metrik: MAE** - Tüm karşılaştırmalar validation MAE üzerinden.

## Proje Yapısı
```
src/config.py    - Tüm hyperparametreler (tek kaynak)
src/dataset.py   - Veri yükleme, KNN imputation, log1p(f3), StandardScaler, PyTorch Dataset
src/model.py     - DNN mimarisi
src/train.py     - Training loop, early stopping, AdamW, CosineAnnealingWarmRestarts
src/evaluate.py  - Metrikler (MAE, RMSE, R2) ve grafikler
src/predict.py   - Test tahmini ve submission üretimi
src/tune.py      - Optuna hyperparameter optimization
Reports/         - Her deney için rapor (zorunlu)
outputs/models/  - Fold başına model ağırlıkları
outputs/plots/   - Eğitim eğrileri ve residual analizi
```

## Veri Pipeline
1. f5'i düşür
2. KNN Impute (k=10) - f1, f4 eksik değerler
3. log1p(clip(f3, lower=0)) - çarpıklık düzeltme
4. StandardScaler - 4 feature'a uygula

## Teknik Kararlar
- SiLU aktivasyon: smooth gradients, non-linear regression için ReLU'dan iyi
- CosineAnnealingWarmRestarts: periyodik LR ile local minima'dan kaçış
- Gradient clipping (max_norm=1.0): eğitim stabilitesi
- Kaiming (He) initialization: ReLU-family için uygun
- AdamW: decoupled weight decay, daha iyi regularization

## Workflow
1. config.py'de değişiklik yap
2. Eğitimi çalıştır, sonuçları gözlemle
3. Reports/ altına rapor yaz (format aşağıda)
4. Önceki sonuçlarla karşılaştır, karar ver

## Rapor Formatı (Reports/XX_deney_adi.md)
```markdown
# Report XX: Deney Adı
**Tarih:** YYYY-MM-DD
**Baseline MAE:** X.XXXX

## Değişiklik
Ne değişti, neden.

## Config
Kullanılan hyperparametreler.

## Sonuçlar
| Fold | Val MAE | Val RMSE | Val R2 |
|------|---------|----------|--------|
| 0    |         |          |        |
| ...  |         |          |        |
| Mean |         |          |        |

## Analiz
Ne işe yaradı, ne yaramadı.

## Sonraki Adımlar
Bu sonuçlara göre ne denenmeli.
```

## W&B (Weights & Biases)
- Proje: `dnn-universal-approx`
- Her eğitimde otomatik loglanır: epoch bazlı loss/MAE/RMSE/R2/LR + CV summary
- `config.py`'de `WANDB_ENABLED = False` ile kapatılabilir
- İlk kullanımda: `wandb login` ile API key gir

## Komutlar
```bash
wandb login              # İlk seferde API key gir
python -m src.train      # Tüm fold'ları eğit (W&B'ye loglar)
python -m src.tune       # Optuna tuning
python -m src.predict    # Submission üret
```
