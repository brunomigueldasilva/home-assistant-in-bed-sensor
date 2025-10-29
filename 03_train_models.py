#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 03_train_models.py
Objective: Train multiple classification models for in bed prediction
Author: Bruno Silva
Date: 2025
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import pandas as pd
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Model training configuration parameters."""
    PROCESSED_DATA_DIR = Path('data_processed')
    MODELS_DIR = Path('models')
    PREDICTIONS_DIR = Path('predictions')
    RANDOM_STATE = 42  # For reproducibility
    OUTPUT_CSV = 'training_times.csv'
    KB_DIVISOR = 1024  # For file size conversion


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_section(title: str, char: str = "=") -> None:
    """
    Print formatted section header.

    Args:
        title: Section title to display
        char: Character to use for border (default: "=")
    """
    print("\n" + char * 80)
    print(title)
    print(char * 80)


def load_pickle(filepath: Path) -> Any:
    """
    Load object from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Any: Unpickled object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj: Any, filepath: Path) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to pickle
        filepath: Path where to save the pickle file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_preprocessed_data(
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load preprocessed train and test data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test

    Raises:
        SystemExit: If data directory or files not found
    """
    print_section("1. LOADING PREPROCESSED DATA")

    # Check if processed data directory exists
    if not Config.PROCESSED_DATA_DIR.exists():
        print(f"✗ ERROR: Directory not found: {Config.PROCESSED_DATA_DIR}")
        print("  Please run 02_preprocessing.py first.")
        exit(1)

    # Load data files
    files_to_load = {
        'X_train': Config.PROCESSED_DATA_DIR / 'X_train.pkl',
        'X_test': Config.PROCESSED_DATA_DIR / 'X_test.pkl',
        'y_train': Config.PROCESSED_DATA_DIR / 'y_train.pkl',
        'y_test': Config.PROCESSED_DATA_DIR / 'y_test.pkl'
    }

    print("Loading files:")
    data = {}
    for name, filepath in files_to_load.items():
        if not filepath.exists():
            print(f"✗ ERROR: File not found: {filepath}")
            exit(1)

        data[name] = load_pickle(filepath)
        print(f"  ✓ {filepath.name:<20} Shape: {data[name].shape}")

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    # Verify data integrity
    print("\n✓ Data loaded successfully!")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Test samples: {X_test.shape[0]:,}")
    print(f"  Number of features: {X_train.shape[1]}")
    print(f"  Target type: {y_train.dtype}")

    # Check class distribution
    print("\nClass distribution:")
    print(f"  Train: {y_train.value_counts().sort_index().to_dict()}")
    print(f"  Test:  {y_test.value_counts().sort_index().to_dict()}")

    return X_train, X_test, y_train, y_test


# ==============================================================================
# SECTION 4: MODEL DEFINITION
# ==============================================================================


def define_models() -> Dict[str, Tuple[Any, str]]:
    """
    Define all models to be trained.

    Returns:
        Dict[str, Tuple[Any, str]]: Dictionary mapping model names to
            (model_object, description) tuples
    """
    print_section("2. DEFINING MODELS")

    """
    MODEL SELECTION RATIONALE:

    We train 5 different algorithms to compare performance and understand
    which approach works best for our sensor-based classification problem.
    Each model has different strengths and assumptions.
    """

    models = {
        'LogisticRegression': (
            LogisticRegression(random_state=Config.RANDOM_STATE, max_iter=1000),
            """Linear model, interpretable, good baseline.
            Assumes linear relationship between features and log-odds of target.
            Fast to train, provides probability estimates.
            Good when: Features have linear relationships with target."""
        ),

        'KNN': (
            KNeighborsClassifier(n_neighbors=5),
            """Non-parametric, instance-based learning.
            Classifies based on majority vote of K nearest neighbors.
            Sensitive to feature scale (that's why we scaled data!).
            Good when: Data has local patterns, decision boundaries are irregular."""
        ),

        'SVM_Linear': (
            SVC(kernel='linear', C=1.0, gamma='scale', random_state=Config.RANDOM_STATE),
            """Finds optimal hyperplane to separate classes.
            Linear kernel: Efficient in high dimensions, works well when classes are linearly separable.
            Robust to outliers through margin maximization.
            Good when: Clear margin of separation exists between classes."""
        ),

        'SVM_RBF': (
            SVC(kernel='rbf', C=1.0, gamma='scale', random_state=Config.RANDOM_STATE),
            """SVM with Radial Basis Function (Gaussian) kernel.
            Allows non-linear decision boundaries by mapping to higher dimensions.
            More flexible than linear SVM, can capture complex patterns.
            Good when: Relationship between features and target is non-linear."""
        ),

        'NaiveBayes': (
            GaussianNB(),
            """Probabilistic classifier based on Bayes theorem.
            Assumes feature independence (naive assumption, often violated but works well).
            Very fast to train, requires small training data.
            Good when: Features are relatively independent, need fast predictions."""
        )
    }

    print(f"Models defined: {len(models)}")
    print("\nModel overview:")
    for i, (name, (model, description)) in enumerate(models.items(), 1):
        print(f"\n{i}. {name}")
        print(f"   Type: {type(model).__name__}")
        # Print first line of description
        first_sentence = description.strip().split('.')[0]
        print(f"   Note: {first_sentence}.")

    return models


# ==============================================================================
# SECTION 5: MODEL TRAINING
# ==============================================================================


def train_models(
    models: Dict[str, Tuple[Any, str]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Train all models and save them along with predictions.

    Args:
        models: Dictionary of models to train
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        pd.DataFrame: DataFrame with training times for each model
    """
    print_section("3. TRAINING MODELS")

    # Create output directories
    Config.MODELS_DIR.mkdir(exist_ok=True)
    Config.PREDICTIONS_DIR.mkdir(exist_ok=True)

    print("Output directories:")
    print(f"  Models: {Config.MODELS_DIR.absolute()}")
    print(f"  Predictions: {Config.PREDICTIONS_DIR.absolute()}")

    # Track training times
    training_times = []

    print(f"\nTraining {len(models)} models...\n")

    for i, (model_name, (model, description)) in enumerate(models.items(), 1):
        print(f"{'─' * 80}")
        print(f"[{i}/{len(models)}] Training: {model_name}")
        print(f"{'─' * 80}")

        # Start timer
        start_time = time.time()

        # Train model
        print(f"  → Fitting model on {X_train.shape[0]:,} training samples...")
        model.fit(X_train, y_train)

        # End timer
        end_time = time.time()
        training_time = end_time - start_time

        print(f"  ✓ Training completed in {training_time:.4f} seconds")

        # Make predictions on train set
        print("  → Making predictions on training set...")
        y_train_pred = model.predict(X_train)

        # Make predictions on test set
        print("  → Making predictions on test set...")
        y_test_pred = model.predict(X_test)

        # Save trained model
        model_filepath = Config.MODELS_DIR / f'model_{model_name}.pkl'
        save_pickle(model, model_filepath)
        print(f"  ✓ Model saved: {model_filepath.name}")

        # Save predictions
        pred_train_filepath = Config.PREDICTIONS_DIR / \
            f'predictions_{model_name}_train.pkl'
        pred_test_filepath = Config.PREDICTIONS_DIR / \
            f'predictions_{model_name}_test.pkl'

        save_pickle(y_train_pred, pred_train_filepath)
        save_pickle(y_test_pred, pred_test_filepath)

        print("  ✓ Predictions saved:")
        print(f"    - {pred_train_filepath.name}")
        print(f"    - {pred_test_filepath.name}")

        # Record training time
        training_times.append({
            'Model': model_name,
            'Training_Time_seconds': training_time
        })

        print()  # Blank line between models

    # Create DataFrame with training times
    times_df = pd.DataFrame(training_times)

    return times_df


# ==============================================================================
# SECTION 6: TRAINING TIMES ANALYSIS
# ==============================================================================


def save_training_times(times_df: pd.DataFrame) -> None:
    """
    Save and display training times.

    Args:
        times_df: DataFrame with training times
    """
    print_section("4. TRAINING TIMES SUMMARY")

    # Sort by training time
    times_df_sorted = times_df.sort_values('Training_Time_seconds')

    # Display table
    print("\nTraining times (sorted by speed):\n")
    print(times_df_sorted.to_string(index=False))

    # Additional statistics
    print("\nStatistics:")
    print(f"  Fastest: {times_df_sorted.iloc[0]['Model']} "
          f"({times_df_sorted.iloc[0]['Training_Time_seconds']:.4f}s)")
    print(f"  Slowest: {times_df_sorted.iloc[-1]['Model']} "
          f"({times_df_sorted.iloc[-1]['Training_Time_seconds']:.4f}s)")
    print(
        f"  Total training time: {
            times_df['Training_Time_seconds'].sum():.4f}s")

    # Save to CSV
    times_df_sorted.to_csv(Config.OUTPUT_CSV, index=False)
    print(f"\n✓ Training times saved: {Config.OUTPUT_CSV}")


# ==============================================================================
# SECTION 7: VERIFICATION
# ==============================================================================


def verify_outputs(models: Dict[str, Tuple[Any, str]]) -> None:
    """
    Verify that all models and predictions were saved correctly.

    Args:
        models: Dictionary of trained models
    """
    print_section("5. VERIFICATION")

    print("Checking saved files...\n")

    # Check models
    print("Models:")
    models_saved = 0
    for model_name in models.keys():
        model_filepath = Config.MODELS_DIR / f'model_{model_name}.pkl'
        if model_filepath.exists():
            file_size = model_filepath.stat().st_size / Config.KB_DIVISOR
            print(f"  ✓ {model_filepath.name:<35} ({file_size:.2f} KB)")
            models_saved += 1
        else:
            print(f"  ✗ {model_filepath.name} - NOT FOUND")

    # Check predictions
    print("\nPredictions:")
    predictions_saved = 0
    for model_name in models.keys():
        for split in ['train', 'test']:
            pred_filepath = Config.PREDICTIONS_DIR / \
                f'predictions_{model_name}_{split}.pkl'
            if pred_filepath.exists():
                file_size = pred_filepath.stat().st_size / Config.KB_DIVISOR
                print(f"  ✓ {pred_filepath.name:<45} ({file_size:.2f} KB)")
                predictions_saved += 1
            else:
                print(f"  ✗ {pred_filepath.name} - NOT FOUND")

    # Summary
    print(f"\n{'─' * 80}")
    print("Verification Summary:")
    print(f"  Models saved: {models_saved}/{len(models)}")
    print(f"  Predictions saved: {predictions_saved}/{len(models) * 2}")

    if models_saved == len(models) and predictions_saved == len(models) * 2:
        print("\n✓ All files saved successfully!")
    else:
        print("\n⚠️  Warning: Some files may be missing")


# ==============================================================================
# SECTION 8: FINAL SUMMARY
# ==============================================================================


def print_final_summary(
        models: Dict[str, Tuple[Any, str]], times_df: pd.DataFrame) -> None:
    """
    Print final training summary.

    Args:
        models: Dictionary of trained models
        times_df: DataFrame with training times
    """
    print_section("TRAINING SUMMARY")

    total_time = times_df['Training_Time_seconds'].sum()

    summary = f"""
✓ Model training completed successfully!

MODELS TRAINED: {len(models)}
  1. Logistic Regression (baseline, linear)
  2. K-Nearest Neighbors (instance-based)
  3. SVM Linear (linear decision boundary)
  4. SVM RBF (non-linear decision boundary)
  5. Naive Bayes (probabilistic)

OUTPUTS:
  - Trained models: {len(models)} files in {Config.MODELS_DIR}/
  - Predictions: {len(models) * 2} files in {Config.PREDICTIONS_DIR}/
  - Training times: {Config.OUTPUT_CSV}
  - Total training time: {total_time:.4f} seconds

FILES STRUCTURE:
  {Config.MODELS_DIR}/
    ├── model_LogisticRegression.pkl
    ├── model_KNN.pkl
    ├── model_SVM_Linear.pkl
    ├── model_SVM_RBF.pkl
    └── model_NaiveBayes.pkl

  {Config.PREDICTIONS_DIR}/
    ├── predictions_LogisticRegression_train.pkl
    ├── predictions_LogisticRegression_test.pkl
    ├── (... and 8 more prediction files)

NEXT STEPS:
  - Run 04_evaluate_models.py to calculate performance metrics
  - Compare models using accuracy, precision, recall, F1-score
  - Analyze confusion matrices and ROC curves
  - Select best model for deployment

NOTE: This script focused only on training and saving models.
      Evaluation metrics will be calculated in the next script.
"""
    print(summary)


# ==============================================================================
# SECTION 9: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """
    Main function that orchestrates the entire training pipeline.

    This function executes the complete training workflow:
    1. Load preprocessed data
    2. Define models
    3. Train all models and save predictions
    4. Save and display training times
    5. Verify all outputs were saved
    6. Print final summary
    """
    print("=" * 80)
    print("MODEL TRAINING - IN BED CLASSIFICATION")
    print("=" * 80)

    # 1. Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data()

    # 2. Define models
    models = define_models()

    # 3. Train all models and save predictions
    times_df = train_models(models, X_train, y_train, X_test, y_test)

    # 4. Save and display training times
    save_training_times(times_df)

    # 5. Verify all outputs were saved
    verify_outputs(models)

    # 6. Print final summary
    print_final_summary(models, times_df)

    print("=" * 80)
    print("TRAINING COMPLETED!")
    print("=" * 80)


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
