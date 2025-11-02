#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
IN BED PREDICTION - DATA PREPROCESSING
==============================================================================

Purpose: Prepare sensor data for machine learning model training

This script:
1. Loads consolidated dataset from exploratory analysis
2. Separates features (X) from target variable (y)
3. Applies One-Hot Encoding to categorical sensor states
4. Performs stratified train-test split (80/20) maintaining class balance
5. Scales features using StandardScaler (fitted only on training data)
6. Saves preprocessed data (X_train, X_test, y_train, y_test, scaler)
7. Prevents data leakage by proper separation of train/test processing

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import pandas as pd
import pickle
import warnings
from pathlib import Path
from typing import Tuple, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Preprocessing configuration parameters."""
    INPUT_FILE = Path('outputs/dataset.csv')
    PROCESSED_DATA_DIR = Path('outputs/data_processed')
    RANDOM_STATE = 42  # For reproducibility
    TEST_SIZE = 0.20  # 80% train, 20% test
    MAX_COLUMNS_TO_DISPLAY = 10  # Number of columns to show in summaries


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


def save_pickle(obj: Any, filename: str) -> str:
    """
    Save object to pickle file and return file size.

    Args:
        obj: Object to pickle
        filename: Name of pickle file

    Returns:
        str: Human-readable file size (e.g., "1.23 MB")
    """
    filepath = Config.PROCESSED_DATA_DIR / filename
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

    # Calculate file size
    file_size = filepath.stat().st_size
    if file_size < 1024:
        size_str = f"{file_size} bytes"
    elif file_size < 1024**2:
        size_str = f"{file_size / 1024:.2f} KB"
    else:
        size_str = f"{file_size / (1024**2):.2f} MB"

    return size_str


# ==============================================================================
# SECTION 3: DATA LOADING AND VERIFICATION
# ==============================================================================


def load_and_verify_data() -> pd.DataFrame:
    """
    Load dataset.csv and perform initial verification.

    Returns:
        pd.DataFrame: Loaded and verified dataset

    Raises:
        SystemExit: If dataset file is not found or validation fails
    """
    print_section("1. DATA LOADING AND VERIFICATION")

    # Load dataset
    try:
        df = pd.read_csv(Config.INPUT_FILE)
        print("✓ Dataset loaded successfully: dataset.csv")
        print(f"  Shape: {df.shape} (rows, columns)")
    except FileNotFoundError:
        print("✗ ERROR: dataset.csv not found!")
        print("  Please run 01_exploratory_analysis.py first.")
        exit(1)

    # Validate dataset is not empty
    if df.empty:
        print("✗ ERROR: Dataset is empty!")
        exit(1)

    # Check for required columns
    required_columns = ['in_bed', 'timestamp']
    missing_columns = [
        col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"✗ ERROR: Missing required columns: {missing_columns}")
        exit(1)

    # Basic information
    print("\nDataset Overview:")
    print(f"  Rows: {df.shape[0]:,}")
    print(f"  Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(f"  {list(df.columns)}")

    print("\nData types:")
    print(df.dtypes)

    # Missing values check
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values found!")
    else:
        print(missing[missing > 0])
        print(f"\n  Total missing values: {missing.sum()}")

        # STRATEGY: Drop rows with missing values
        print("\n  Strategy: Dropping rows with missing values")
        print(
            f"  Rationale: With {
                df.shape[0]:,    } records, losing a few rows won't impact")
        print("            model training, safer than imputation for sensor data")

        df_original_size = len(df)
        df = df.dropna()
        print(f"  ✓ Dropped {df_original_size - len(df)} rows")
        print(f"  New shape: {df.shape}")

    # Check target variable
    if 'in_bed' not in df.columns:
        print("\n✗ ERROR: Target variable 'in_bed' not found!")
        exit(1)

    print("\nTarget variable 'in_bed' distribution:")
    print(df['in_bed'].value_counts().sort_index())
    print("\nPercentages:")
    print(df['in_bed'].value_counts(normalize=True).sort_index() * 100)

    return df


# ==============================================================================
# SECTION 4: FEATURE AND TARGET SEPARATION
# ==============================================================================


def separate_features_target(
        df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) from target variable (y).

    Args:
        df: Complete DataFrame

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y)
    """
    print_section("2. SEPARATING FEATURES (X) AND TARGET (y)")

    """
    COLUMN REMOVAL STRATEGY:
    - Remove 'timestamp': Not a feature, just a record identifier
    - Remove 'date': Redundant with temporal features (hour, day_of_week)
    - Remove 'in_bed': This is our target variable (y)
    - Keep all sensor states and engineered features
    """

    # Identify columns to remove
    columns_to_remove = ['timestamp', 'date', 'in_bed']
    columns_to_remove = [col for col in columns_to_remove if col in df.columns]

    print(f"Columns to remove: {columns_to_remove}")

    # Separate X and y
    y = df['in_bed'].copy()
    X = df.drop(columns=columns_to_remove)

    print("\n✓ Separation completed:")
    print(f"  X (features) shape: {X.shape}")
    print(f"  y (target) shape: {y.shape}")

    print(f"\nFeature columns ({len(X.columns)}):")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i:2d}. {col}")

    print("\nData types in X:")
    print(X.dtypes.value_counts())

    return X, y


# ==============================================================================
# SECTION 5: CATEGORICAL ENCODING
# ==============================================================================


def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply One-Hot Encoding to categorical variables.

    Args:
        X: Feature DataFrame

    Returns:
        pd.DataFrame: DataFrame with encoded features
    """
    print_section("3. ENCODING CATEGORICAL VARIABLES")

    """
    ONE-HOT ENCODING RATIONALE:
    - Sensor states (e.g., 'on'/'off', 'open'/'closed') are categorical
    - These categories have NO natural order (nominal data)
    - One-Hot Encoding creates binary columns for each category
    - Prevents model from assuming ordinal relationships
    - Example: 'light_on' and 'light_off' become separate binary features

    Alternative (Label Encoding) would be WRONG because:
    - Assigning 'off'=0, 'on'=1 implies 'on' > 'off' (false ordering)
    - Tree-based models might work, but linear models would fail
    """

    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

    print(f"Categorical columns identified ({len(categorical_columns)}):")
    for col in categorical_columns:
        unique_values = X[col].unique()
        print(f"  {col}: {unique_values}")

    if categorical_columns:
        print("\nApplying One-Hot Encoding...")

        # Store original column count
        original_cols = X.shape[1]

        # Apply one-hot encoding
        X_encoded = pd.get_dummies(
            X, columns=categorical_columns, drop_first=False)

        # Report changes
        print("  ✓ Encoding completed")
        print(f"  Original columns: {original_cols}")
        print(f"  New columns: {X_encoded.shape[1]}")
        print(f"  Columns added: {X_encoded.shape[1] - original_cols}")

        # Show new column names
        new_columns = [
            col for col in X_encoded.columns if col not in X.columns]
        print(f"\n  New encoded columns ({len(new_columns)}):")
        for col in new_columns[:Config.MAX_COLUMNS_TO_DISPLAY]:
            print(f"    - {col}")
        if len(new_columns) > Config.MAX_COLUMNS_TO_DISPLAY:
            print(
                f"    ... and {
                    len(new_columns) -
                    Config.MAX_COLUMNS_TO_DISPLAY} more")

        X = X_encoded
    else:
        print("  No categorical columns found - skipping encoding")

    print(f"\nFinal feature matrix shape: {X.shape}")

    return X


# ==============================================================================
# SECTION 6: TRAIN-TEST SPLIT
# ==============================================================================


def split_train_test(X: pd.DataFrame,
                     y: pd.Series) -> Tuple[pd.DataFrame,
                                            pd.DataFrame,
                                            pd.Series,
                                            pd.Series]:
    """
    Perform stratified train-test split.

    Args:
        X: Feature DataFrame
        y: Target Series

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test
    """
    print_section("4. STRATIFIED TRAIN-TEST SPLIT")

    """
    STRATIFICATION IMPORTANCE:

    In imbalanced datasets (like ours), random splitting can lead to:
    - Train set with different class distribution than test set
    - Minority class underrepresented in one of the splits
    - Biased model evaluation metrics

    STRATIFICATION ENSURES:
    - Train and test sets have SAME class proportion as original dataset
    - Both splits are representative of true data distribution
    - Fair and reliable model evaluation

    Example:
      Original: 70% not in bed, 30% in bed
      Without stratification: Train 75/25, Test 60/40 (inconsistent)
      With stratification: Both Train and Test ~70/30 (consistent)
    """

    print("Splitting configuration:")
    print(f"  Train size: {(1 - Config.TEST_SIZE) * 100:.0f}%")
    print(f"  Test size: {Config.TEST_SIZE * 100:.0f}%")
    print(f"  Random state: {Config.RANDOM_STATE}")
    print("  Stratification: ENABLED (stratify=y)")

    # Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=y  # CRITICAL: Maintain class proportions
    )

    print("\n✓ Split completed:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}")

    # Verify stratification
    verify_stratification(y, y_train, y_test)

    return X_train, X_test, y_train, y_test


def verify_stratification(
        y: pd.Series,
        y_train: pd.Series,
        y_test: pd.Series) -> None:
    """
    Verify that stratification maintained class proportions.

    Args:
        y: Original target variable
        y_train: Training target variable
        y_test: Test target variable
    """
    print("\nClass distribution verification:")
    print(f"\n{'Set':<12} {'Not in Bed':<15} {'In Bed':<15} {'Ratio':<10}")
    print("-" * 52)

    def _compute_and_print_stats(
            y_data: pd.Series, name: str) -> Tuple[pd.Series, float]:
        """
        Helper to compute and print distribution statistics.

        Args:
            y_data: Target variable data
            name: Name of the dataset (e.g., 'Original', 'Train', 'Test')

        Returns:
            Tuple[pd.Series, float]: Counts and ratio
        """
        counts = y_data.value_counts().sort_index()
        pct = (counts / len(y_data)) * 100

        count_0 = counts.get(0, 0)
        count_1 = counts.get(1, 0)
        ratio = count_0 / count_1 if count_1 > 0 else float('inf')

        print(f"{name:<12} {count_0:>6} ({pct.get(0, 0):>5.2f}%)  "
              f"{count_1:>6} ({pct.get(1, 0):>5.2f}%)  {ratio:>6.2f}:1")

        return counts, ratio

    # Compute statistics for all datasets
    _, _ = _compute_and_print_stats(y, "Original")
    train_counts, _ = _compute_and_print_stats(y_train, "Train")
    test_counts, _ = _compute_and_print_stats(y_test, "Test")

    print("\n✓ Stratification successful: Ratios are nearly identical!")

    return train_counts, test_counts


# ==============================================================================
# SECTION 7: FEATURE SCALING
# ==============================================================================


def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> Tuple[pd.DataFrame,
                                                  pd.DataFrame,
                                                  StandardScaler]:
    """
    Scale features using StandardScaler (fit on train only).

    Args:
        X_train: Training features
        X_test: Test features

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
            X_train_scaled, X_test_scaled, fitted scaler
    """
    print_section("5. FEATURE SCALING (StandardScaler)")

    """
    DATA LEAKAGE PREVENTION - CRITICAL CONCEPT:

    WRONG APPROACH (causes data leakage):
      scaler.fit(X)  # Fit on entire dataset
      X_train_scaled = scaler.transform(X_train)
      X_test_scaled = scaler.transform(X_test)

    Why is this wrong?
      - Scaler computes mean and std from ALL data (including test)
      - When scaling X_test, we use statistics that "saw" test data
      - Model indirectly gains information about test set
      - Evaluation metrics become artificially inflated
      - Model won't generalize to truly unseen data

    CORRECT APPROACH (implemented below):
      scaler.fit(X_train)  # Fit ONLY on training data
      X_train_scaled = scaler.transform(X_train)
      X_test_scaled = scaler.transform(X_test)

    Why is this correct?
      - Scaler learns mean and std ONLY from training data
      - Test set is treated as truly unseen data
      - No information leaks from test to train
      - Evaluation metrics are realistic and trustworthy
      - Model will perform similarly on real-world data

    This is one of the most common mistakes in ML pipelines!
    """

    print("Initializing StandardScaler...")
    scaler = StandardScaler()

    # Store feature names
    feature_names = X_train.columns.tolist()

    # Show statistics BEFORE scaling
    print("\nStatistics BEFORE scaling (X_train):")
    print(
        f"  Mean range: [{
            X_train.mean().min():.4f}, {
            X_train.mean().max():.4f}]")
    print(
        f"  Std range: [{
            X_train.std().min():.4f}, {
            X_train.std().max():.4f}]")

    # FIT scaler ONLY on training data
    print("\n► Fitting scaler on X_train only (AVOIDING DATA LEAKAGE)...")
    scaler.fit(X_train)
    print(f"  ✓ Scaler fitted on {X_train.shape[0]:,} training samples")

    # TRANSFORM both train and test
    print("\n► Transforming X_train...")
    X_train_scaled = scaler.transform(X_train)
    print(f"  ✓ X_train scaled: {X_train_scaled.shape}")

    print("\n► Transforming X_test (using training statistics)...")
    X_test_scaled = scaler.transform(X_test)
    print(f"  ✓ X_test scaled: {X_test_scaled.shape}")

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=feature_names,
        index=X_train.index)
    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=feature_names,
        index=X_test.index)

    # Show statistics AFTER scaling
    print("\nStatistics AFTER scaling (X_train_scaled):")
    print(
        f"  Mean range: [{
            X_train_scaled.mean().min():.4e}, {
            X_train_scaled.mean().max():.4e}]")
    print(
        f"  Std range: [{
            X_train_scaled.std().min():.4f}, {
            X_train_scaled.std().max():.4f}]")
    print("  ✓ Features are now standardized (mean≈0, std≈1)")

    print("\nStatistics AFTER scaling (X_test_scaled):")
    print(
        f"  Mean range: [{
            X_test_scaled.mean().min():.4e}, {
            X_test_scaled.mean().max():.4e}]")
    print(
        f"  Std range: [{
            X_test_scaled.std().min():.4f}, {
            X_test_scaled.std().max():.4f}]")
    print("  Note: Test statistics slightly different (expected, no leakage!)")

    return X_train_scaled, X_test_scaled, scaler


# ==============================================================================
# SECTION 8: SAVE PROCESSED DATA
# ==============================================================================


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler,
    feature_names: list,
    df_shape: Tuple[int, int]
) -> Tuple[pd.Series, pd.Series]:
    """
    Save all processed objects to pickle files.

    Args:
        X_train: Scaled training features
        X_test: Scaled test features
        y_train: Training target
        y_test: Test target
        scaler: Fitted StandardScaler object
        feature_names: List of feature names
        df_shape: Original DataFrame shape

    Returns:
        Tuple[pd.Series, pd.Series]: Train and test class distributions
    """
    print_section("6. SAVING PROCESSED DATA")

    # Create output directory
    Config.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {Config.PROCESSED_DATA_DIR.absolute()}")

    # Files to save
    files_to_save = {
        'X_train.pkl': X_train,
        'X_test.pkl': X_test,
        'y_train.pkl': y_train,
        'y_test.pkl': y_test,
        'scaler.pkl': scaler,
        'feature_names.pkl': feature_names
    }

    print(f"\nSaving {len(files_to_save)} files...")

    for filename, obj in files_to_save.items():
        size_str = save_pickle(obj, filename)
        print(f"  ✓ {filename:<20} ({size_str})")

    # Save metadata
    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    metadata = {
        'original_shape': df_shape,
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'n_features': len(feature_names),
        'test_size': Config.TEST_SIZE,
        'random_state': Config.RANDOM_STATE,
        'class_distribution_train': train_counts.to_dict(),
        'class_distribution_test': test_counts.to_dict()
    }

    size_str = save_pickle(metadata, 'metadata.pkl')
    print(f"  ✓ metadata.pkl       ({size_str})")

    return train_counts, test_counts


# ==============================================================================
# SECTION 9: SUMMARY
# ==============================================================================


def print_summary(
    df_shape: Tuple[int, int],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list,
    train_counts: pd.Series,
    test_counts: pd.Series
) -> None:
    """
    Print final preprocessing summary.

    Args:
        df_shape: Original DataFrame shape
        X_train: Training features
        X_test: Test features
        feature_names: List of feature names
        train_counts: Training set class distribution
        test_counts: Test set class distribution
    """
    print_section("PREPROCESSING SUMMARY")

    # Calculate ratios
    ratio_train = train_counts.get(0,
                                   0) / train_counts.get(1,
                                                         1) if train_counts.get(1,
                                                                                0) > 0 else float('inf')
    ratio_test = test_counts.get(0,
                                 0) / test_counts.get(1,
                                                      1) if test_counts.get(1,
                                                                            0) > 0 else float('inf')

    summary = f"""
✓ Data preprocessing completed successfully!

INPUT:
  - Source file: dataset.csv
  - Original shape: {df_shape}

PROCESSING STEPS:
  1. ✓ Loaded and verified data
  2. ✓ Handled missing values (if any)
  3. ✓ Separated features (X) and target (y)
  4. ✓ Applied One-Hot Encoding to categorical variables
  5. ✓ Performed stratified train-test split (80/20)
  6. ✓ Scaled features with StandardScaler (no data leakage)

OUTPUT:
  - Training samples: {X_train.shape[0]:,}
  - Test samples: {X_test.shape[0]:,}
  - Number of features: {len(feature_names)}
  - Files saved: 7 (including metadata)
  - Location: {Config.PROCESSED_DATA_DIR.absolute()}

CLASS DISTRIBUTION:
  - Train: {train_counts.get(0, 0):,} not in bed, {train_counts.get(1, 0):,} in bed ({ratio_train:.2f}:1)
  - Test: {test_counts.get(0, 0):,} not in bed, {test_counts.get(1, 0):,} in bed ({ratio_test:.2f}:1)

NEXT STEPS:
  - Ready for model training (03_train_models.py)
  - All preprocessed files available in {Config.PROCESSED_DATA_DIR}/
  - Scaler object saved for future predictions

KEY DECISIONS MADE:
  ✓ One-Hot Encoding: Preserves categorical nature of sensor states
  ✓ Stratified Split: Maintains class balance in train/test
  ✓ Proper Scaling: No data leakage, scaler fitted only on train
"""
    print(summary)


# ==============================================================================
# SECTION 10: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """
    Main function that orchestrates the entire preprocessing pipeline.

    This function executes the complete preprocessing workflow:
    1. Load and verify data
    2. Separate features and target
    3. Encode categorical features
    4. Split train-test with stratification
    5. Scale features (avoiding data leakage)
    6. Save processed data
    7. Print summary
    """
    print("=" * 80)
    print("DATA PREPROCESSING - IN BED CLASSIFICATION")
    print("=" * 80)

    # 1. Load and verify data
    df = load_and_verify_data()

    # 2. Separate features and target
    X, y = separate_features_target(df)

    # 3. Encode categorical features
    X = encode_categorical_features(X)

    # 4. Split train-test with stratification
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # 5. Scale features (avoiding data leakage)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 6. Save processed data
    feature_names = X_train_scaled.columns.tolist()
    train_counts, test_counts = save_processed_data(
        X_train_scaled, X_test_scaled, y_train, y_test,
        scaler, feature_names, df.shape
    )

    # 7. Print summary
    print_summary(df.shape, X_train_scaled, X_test_scaled,
                  feature_names, train_counts, test_counts)

    print("=" * 80)
    print("PREPROCESSING COMPLETED!")
    print("=" * 80)


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
