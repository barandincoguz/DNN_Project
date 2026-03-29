"""
Test seti prediction ve Kaggle submission oluşturma.
"""
import os
import pandas as pd
from src.config import TEST_PATH, SUBMISSION_PATH
from src.train import train_all_folds
from src.evaluate import plot_training_curves


def generate_submission(test_preds, test_df):
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission = pd.DataFrame({
        'id': test_df['id'],
        'target': test_preds
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSubmission kaydedildi: {SUBMISSION_PATH}")
    print(f"  Shape: {submission.shape}")
    print(f"  Target: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")
    return submission


def main():
    fold_maes, test_preds, histories = train_all_folds()
    plot_training_curves(histories)

    test_df = pd.read_csv(TEST_PATH)
    return generate_submission(test_preds, test_df)


if __name__ == '__main__':
    main()
