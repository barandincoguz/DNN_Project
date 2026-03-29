# Deep Neural Network — Universal Function Approximation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/barandincoguz/DNN_Project/blob/main/notebooks/colab_setup.ipynb)

Sentetik olarak üretilmiş 4 feature'dan non-lineer bir target değişkenini tahmin eden **tek bir Deep Neural Network** modeli. Yarışma formatı: profesörün gizli test setinde en düşük MAE kazanır.

---

## Hızlı Başlangıç

**Colab'da çalıştır:** Yukarıdaki badge'e tıkla → Runtime > GPU seç → Hücreleri sırayla çalıştır.

**Yerel:**
```bash
pip install -r requirements.txt
python -m src.train      # Eğitim (5-fold CV)
python -m src.predict    # Submission üret
```

---

## EDA Bulguları

| Feature | Aralık | Rolü | Karar |
|---------|--------|------|-------|
| f1 | [-9.9, 6.3] | Non-lineer katkı | KULLAN |
| f2 | [-25, 30] | **Ana driver** (r:0.58) | KULLAN |
| f3 | [0.2, 54.3] | Hafif katkı, çarpık (skew:3.51) | log1p uygula |
| f4 | [0.6, 978] | Etkileşim yoluyla katkı | KULLAN |
| f5 | [-6.3, 6.9] | **Tuzak** — çıkarınca MAE düşüyor | DROP |

**Eksik veri:** f1 (41 train, 10 test), f4 (30 train, 10 test) → KNN Imputer (k=10)

---

## Pipeline

```
Raw Data → f5 DROP → KNN Impute (k=10) → log1p(f3) → StandardScaler → DNN → Prediction
```

---

## Proje Yapısı

```
src/
  config.py     Tüm hyperparametreler (tek kaynak)
  dataset.py    Veri yükleme, imputation, scaling, PyTorch Dataset
  model.py      DNN mimarisi
  train.py      5-fold CV eğitim + W&B logging
  evaluate.py   MAE, RMSE, R2 metrikleri + grafikler
  predict.py    Submission üretimi
  tune.py       Optuna hyperparameter arama

notebooks/
  colab_setup.ipynb   Colab'da çalıştırılacak ana notebook

data/           train.csv, test.csv
outputs/        models/, plots/, submission.csv
Reports/        Deney raporları
```

---

## Teknik Kararlar

| Bileşen | Seçim | Neden |
|---------|-------|-------|
| Aktivasyon | SiLU | Smooth gradients, non-linear regression için ReLU'dan iyi |
| Optimizer | AdamW | Decoupled weight decay |
| LR Schedule | CosineAnnealingWarmRestarts | Periyodik keşif, local minima'dan kaçış |
| Regularization | Dropout + BatchNorm + Weight Decay | 1600 sample → overfitting riski yüksek |
| Validation | 5-Fold CV | Güvenilir performans tahmini |
| Metrik | MAE (primary) | Yarışma kriteri |
| Init | Kaiming (He) | SiLU/ReLU ailesi için standart |
| Tracking | Weights & Biases | Deney metrikleri karşılaştırması |

---

## Colab Workflow

1. **Kurulum** — Repo clone, pip install, GPU kontrol, W&B login
2. **EDA** — Veri analizi (ilk seferde)
3. **Config** — Hyperparameter override
4. **Eğitim + Submission** — Tek hücre: eğitir → grafik → submission.csv
5. **Sonuçlar** — Grafikleri gör, submission indir
6. **Push** — Sonuçları GitHub'a gönder
