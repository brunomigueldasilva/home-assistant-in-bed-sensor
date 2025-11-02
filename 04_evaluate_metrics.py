#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
IN BED PREDICTION - METRICS EVALUATION
==============================================================================

Purpose: Calculate and compare performance metrics for all trained models

This script:
1. Loads ground truth labels (y_test) and model predictions
2. Calculates 6 key metrics (Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC)
3. Compares all models in a comparative table
4. Identifies best performing model by F1-Score
5. Analyzes metric trade-offs (Precision vs Recall)
6. Discusses why accuracy alone is insufficient for imbalanced data
7. Saves comparative metrics to CSV and Markdown formats

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
from typing import Dict, Tuple, Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

warnings.filterwarnings('ignore')


# Configuration Constants
class Config:
    """Evaluation configuration parameters."""
    PROCESSED_DATA_DIR = Path('outputs/data_processed')
    MODELS_DIR = Path('outputs/models')
    PREDICTIONS_DIR = Path('outputs/predictions')

    # Model names (must match training script)
    MODEL_NAMES = [
        'LogisticRegression',
        'KNN',
        'SVM_Linear',
        'SVM_RBF',
        'NaiveBayes'
    ]

    # Output files
    OUTPUT_CSV = 'outputs/comparative_metrics.csv'
    OUTPUT_MD = 'outputs/comparative_metrics.md'

    # Thresholds
    LOW_RECALL_THRESHOLD = 0.3  # Models below this may be biased


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


# ==============================================================================
# SECTION 3: DATA LOADING
# ==============================================================================


def load_ground_truth() -> Tuple[pd.Series, pd.Series, float]:
    """
    Load true labels for train and test sets.

    Returns:
        Tuple[pd.Series, pd.Series, float]: y_train, y_test, baseline_accuracy

    Raises:
        SystemExit: If ground truth files not found
    """
    print_section("1. LOADING GROUND TRUTH LABELS")

    # Load true labels
    y_train_path = Config.PROCESSED_DATA_DIR / 'y_train.pkl'
    y_test_path = Config.PROCESSED_DATA_DIR / 'y_test.pkl'

    if not y_train_path.exists() or not y_test_path.exists():
        print("✗ ERROR: Ground truth labels not found!")
        print("  Please run 02_preprocessing.py first.")
        exit(1)

    y_train = load_pickle(y_train_path)
    y_test = load_pickle(y_test_path)

    print("✓ Ground truth loaded:")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}")

    # Show class distribution
    print("\nClass distribution in test set:")
    test_counts = y_test.value_counts().sort_index()
    test_pct = y_test.value_counts(normalize=True).sort_index() * 100

    print(
        f"  Class 0 (not in bed): {
            test_counts.get(
                0, 0):,} samples ({
            test_pct.get(
                0, 0):.2f}%)")
    print(
        f"  Class 1 (in bed):     {
            test_counts.get(
                1, 0):,} samples ({
            test_pct.get(
                1, 0):.2f}%)")

    # Calculate baseline accuracy (always predict majority class)
    baseline_accuracy = test_pct.max() / 100
    print(
        f"\n  Baseline accuracy (always predict majority): {
            baseline_accuracy:.4f}")
    print("  → Any model below this is just learning the majority class!")

    return y_train, y_test, baseline_accuracy


def load_all_predictions() -> Dict[str, Dict[str, Any]]:
    """
    Load predictions from all trained models.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping model names to
            {'train': y_pred_train, 'test': y_pred_test}

    Raises:
        SystemExit: If predictions directory not found
    """
    print_section("2. LOADING MODEL PREDICTIONS")

    if not Config.PREDICTIONS_DIR.exists():
        print("✗ ERROR: Predictions directory not found!")
        print("  Please run 03_train_models.py first.")
        exit(1)

    predictions = {}

    print("Loading predictions for each model:\n")

    for model_name in Config.MODEL_NAMES:
        train_pred_path = Config.PREDICTIONS_DIR / \
            f'predictions_{model_name}_train.pkl'
        test_pred_path = Config.PREDICTIONS_DIR / \
            f'predictions_{model_name}_test.pkl'

        if not train_pred_path.exists() or not test_pred_path.exists():
            print(f"⚠️  Warning: Predictions not found for {model_name}")
            continue

        y_pred_train = load_pickle(train_pred_path)
        y_pred_test = load_pickle(test_pred_path)

        predictions[model_name] = {
            'train': y_pred_train,
            'test': y_pred_test
        }

        print(
            f"  ✓ {
                model_name:<25} Train: {
                y_pred_train.shape}  Test: {
                y_pred_test.shape}")

    print(f"\n✓ Loaded predictions for {len(predictions)} models")

    return predictions


# ==============================================================================
# SECTION 4: METRIC CALCULATION FUNCTIONS
# ==============================================================================


def calculate_specificity(y_true: pd.Series, y_pred: Any) -> float:
    """
    Calculate Specificity (True Negative Rate).

    Specificity = TN / (TN + FP)

    Measures: Of all actual negatives, how many did we correctly identify?

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        float: Specificity score
    """
    cm = confusion_matrix(y_true, y_pred)

    # Confusion matrix structure:
    #           Predicted
    #           0    1
    # Actual 0  TN   FP
    #        1  FN   TP

    tn = cm[0, 0]
    fp = cm[0, 1]

    if (tn + fp) == 0:
        return 0.0

    specificity = tn / (tn + fp)
    return specificity


def calculate_model_metrics(
        y_true: pd.Series, y_pred: Any, model_name: str) -> Dict[str, Any]:
    """
    Calculate all evaluation metrics for a single model.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model (for potential probability scores)

    Returns:
        Dict[str, Any]: Dictionary with all metrics
    """
    """
    METRIC INTERPRETATION IN THIS CONTEXT:

    1. ACCURACY: Overall correctness
       - Percentage of correct predictions (both classes)
       - ⚠️ MISLEADING in imbalanced data!
       - Example: 90% class 0, 10% class 1 → Always predict 0 = 90% accuracy

    2. PRECISION (Positive Predictive Value):
       - Of all "in bed" predictions, how many were correct?
       - Precision = TP / (TP + FP)
       - High precision → Few false alarms
       - Important when: Cost of False Positives is high
       - Our case: Saying "in bed" when not → Could trigger actions wrongly

    3. RECALL (Sensitivity, True Positive Rate):
       - Of all actual "in bed" cases, how many did we detect?
       - Recall = TP / (TP + FN)
       - High recall → Few missed detections
       - Important when: Cost of False Negatives is high
       - Our case: Missing "in bed" → Could fail to trigger needed actions

    4. SPECIFICITY (True Negative Rate):
       - Of all actual "not in bed" cases, how many did we correctly identify?
       - Specificity = TN / (TN + FP)
       - High specificity → Few false positives for negative class
       - Important when: Need to correctly identify negative cases

    5. F1-SCORE:
       - Harmonic mean of Precision and Recall
       - F1 = 2 * (Precision * Recall) / (Precision + Recall)
       - Balances both concerns equally
       - Good when: Both FP and FN are important
       - Range: 0 to 1 (higher is better)

    6. ROC-AUC (Area Under ROC Curve):
       - Measures discrimination ability across all thresholds
       - AUC = 1.0: Perfect classifier
       - AUC = 0.5: Random guessing (coin flip)
       - AUC < 0.5: Worse than random (predictions inverted)
       - Robust to class imbalance

    PRECISION vs RECALL TRADE-OFF:
    - These metrics are inversely related
    - Increasing threshold (e.g., 0.5 → 0.7):
      → More confident "in bed" predictions → Higher Precision, Lower Recall
    - Decreasing threshold (e.g., 0.5 → 0.3):
      → More "in bed" predictions → Lower Precision, Higher Recall

    WHICH IS MORE IMPORTANT?
    Depends on the application:
    - Home automation (lights, heating): FN might be worse (miss comfort needs)
    - Security system: FP might be worse (false alarms annoying)
    - Medical diagnosis: Usually prioritize Recall (don't miss sick patients)
    """

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = calculate_specificity(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC calculation
    try:
        # Try to get probability scores for better ROC-AUC
        # (This would require loading the actual model, not just predictions)
        # For now, we use the binary predictions
        roc_auc = roc_auc_score(y_true, y_pred)
    except Exception:
        # If ROC-AUC fails (rare), set to None
        roc_auc = None

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

    return metrics


# ==============================================================================
# SECTION 5: METRICS CALCULATION FOR ALL MODELS
# ==============================================================================


def calculate_all_metrics(
        predictions: Dict[str, Dict[str, Any]], y_test: pd.Series) -> pd.DataFrame:
    """
    Calculate metrics for all models on test set.

    Args:
        predictions: Dictionary with model predictions
        y_test: True test labels

    Returns:
        pd.DataFrame: DataFrame with metrics for all models
    """
    print_section("3. CALCULATING EVALUATION METRICS (TEST SET)")

    """
    WHY ONLY TEST SET?

    Training metrics can be misleading:
    - Models can "memorize" training data (overfitting)
    - High training performance doesn't guarantee generalization
    - Test set represents unseen, real-world data

    Test metrics show true model performance on new data.
    """

    print("\nCalculating metrics for each model...\n")

    all_metrics = []

    for i, (model_name, preds) in enumerate(predictions.items(), 1):
        print(f"[{i}/{len(predictions)}] {model_name}")

        y_pred_test = preds['test']

        # Calculate metrics
        metrics = calculate_model_metrics(y_test, y_pred_test, model_name)
        all_metrics.append(metrics)

        # Display metrics for this model
        print(f"  Accuracy:    {metrics['Accuracy']:.4f}")
        print(f"  Precision:   {metrics['Precision']:.4f}")
        print(f"  Recall:      {metrics['Recall']:.4f}")
        print(f"  Specificity: {metrics['Specificity']:.4f}")
        print(f"  F1-Score:    {metrics['F1-Score']:.4f}")
        if metrics['ROC-AUC'] is not None:
            print(f"  ROC-AUC:     {metrics['ROC-AUC']:.4f}")
        print()

    # Create DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    return metrics_df


# ==============================================================================
# SECTION 6: COMPARATIVE ANALYSIS
# ==============================================================================


def create_comparative_table(
        metrics_df: pd.DataFrame,
        baseline_accuracy: float) -> pd.DataFrame:
    """
    Create and format comparative metrics table.

    Args:
        metrics_df: DataFrame with metrics
        baseline_accuracy: Baseline accuracy (majority class)

    Returns:
        pd.DataFrame: Formatted DataFrame sorted by F1-Score
    """
    print_section("4. COMPARATIVE ANALYSIS")

    # Sort by F1-Score (descending)
    metrics_df_sorted = metrics_df.sort_values(
        'F1-Score',
        ascending=False).reset_index(
        drop=True)

    print("\nMetrics Comparison (sorted by F1-Score):\n")

    # Format for display
    display_df = metrics_df_sorted.copy()
    for col in [
        'Accuracy',
        'Precision',
        'Recall',
        'Specificity',
        'F1-Score',
            'ROC-AUC']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    print(display_df.to_string(index=False))

    # Identify best model
    best_model = metrics_df_sorted.iloc[0]

    print(f"\n{'=' * 80}")
    print(f"🏆 BEST MODEL: {best_model['Model']}")
    print(f"{'=' * 80}")
    print(f"  F1-Score:    {best_model['F1-Score']:.4f} ⭐")
    print(f"  Accuracy:    {best_model['Accuracy']:.4f}")
    print(f"  Precision:   {best_model['Precision']:.4f}")
    print(f"  Recall:      {best_model['Recall']:.4f}")
    print(f"  Specificity: {best_model['Specificity']:.4f}")
    if pd.notna(best_model['ROC-AUC']):
        print(f"  ROC-AUC:     {best_model['ROC-AUC']:.4f}")

    # Compare with baseline
    print(f"\n{'─' * 80}")
    print("Comparison with Baseline:")
    print(f"  Baseline (always predict majority): {baseline_accuracy:.4f}")
    print(
        f"  Best model accuracy:                 {
            best_model['Accuracy']:.4f}")

    if best_model['Accuracy'] > baseline_accuracy:
        improvement = (
            (best_model['Accuracy'] - baseline_accuracy) / baseline_accuracy) * 100
        print(f"  ✓ Improvement: +{improvement:.2f}%")
    else:
        print("  ⚠️ Warning: Best model is not better than baseline!")

    # Analysis of all models vs baseline
    print(f"\n{'─' * 80}")
    print("All models vs Baseline:")
    for _, row in metrics_df_sorted.iterrows():
        status = "✓" if row['Accuracy'] > baseline_accuracy else "✗"
        print(f"  {status} {row['Model']:<25} Accuracy: {row['Accuracy']:.4f}")

    return metrics_df_sorted


# ==============================================================================
# SECTION 7: DETAILED DISCUSSION
# ==============================================================================


def print_detailed_discussion(metrics_df: pd.DataFrame) -> None:
    """
    Print detailed interpretation and discussion of results.

    Args:
        metrics_df: DataFrame with sorted metrics
    """
    print_section("5. DETAILED INTERPRETATION")

    """
    CRITICAL UNDERSTANDING: WHY ACCURACY IS NOT ENOUGH

    Imagine a dataset with 95% class 0 (not in bed) and 5% class 1 (in bed).

    A "dumb" model that ALWAYS predicts class 0:
    - Accuracy: 95% (looks great!)
    - Recall: 0% (missed ALL "in bed" cases)
    - Precision: undefined (never predicted "in bed")
    - F1-Score: 0% (terrible)

    This model is useless but has high accuracy!

    WHY DOES THIS HAPPEN?
    - Accuracy treats all errors equally: FP and FN weighted the same
    - In imbalanced data, minority class contributes little to accuracy
    - A model can ignore minority class and still get high accuracy

    BETTER METRICS FOR IMBALANCED DATA:
    1. F1-Score: Balances Precision and Recall
    2. ROC-AUC: Evaluates across all thresholds
    3. Precision-Recall for minority class
    4. Confusion Matrix: Shows exactly what errors occur
    """

    print("\n📊 KEY INSIGHTS:\n")

    # Find model with highest and lowest metrics
    best_f1_model = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
    best_recall_model = metrics_df.loc[metrics_df['Recall'].idxmax()]
    best_precision_model = metrics_df.loc[metrics_df['Precision'].idxmax()]

    print("1. BEST F1-SCORE:")
    print(f"   {best_f1_model['Model']}: {best_f1_model['F1-Score']:.4f}")
    print("   → Best overall balance between Precision and Recall")
    print("   → Recommended for general use\n")

    print("2. BEST RECALL (Sensitivity):")
    print(
        f"   {
            best_recall_model['Model']}: {
            best_recall_model['Recall']:.4f}")
    print("   → Best at detecting 'in bed' cases (fewest missed detections)")
    print("   → Use when: Missing 'in bed' is costly (e.g., safety monitoring)\n")

    print("3. BEST PRECISION:")
    print(
        f"   {
            best_precision_model['Model']}: {
            best_precision_model['Precision']:.4f}")
    print("   → Most confident when predicting 'in bed'")
    print("   → Use when: False alarms are costly (e.g., unnecessary actions)\n")

    print("4. TRADE-OFF ANALYSIS:")
    print("   Precision ↑ → Fewer false positives (more conservative)")
    print("   Recall ↑    → Fewer false negatives (more aggressive)")
    print("   Both can't be maximized simultaneously!")
    print("   F1-Score finds the sweet spot\n")

    print("5. SPECIFICITY IMPORTANCE:")
    print("   High Specificity → Correctly identifies 'not in bed'")
    print("   Important to avoid triggering actions when not needed")
    print("   Should be balanced with Recall\n")

    # Check if any model is just learning majority class
    print("6. SANITY CHECK:")
    poor_performers = metrics_df[metrics_df['Recall']
                                 < Config.LOW_RECALL_THRESHOLD]
    if len(poor_performers) > 0:
        print(
            f"   ⚠️ Models with low Recall (<{
                Config.LOW_RECALL_THRESHOLD *
                100:.0f}%):")
        for _, row in poor_performers.iterrows():
            print(
                f"     - {row['Model']}: May be biased toward majority class")
    else:
        print(
            f"   ✓ All models have reasonable Recall (>{
                Config.LOW_RECALL_THRESHOLD *
                100:.0f}%)")
        print("   ✓ No model is ignoring the minority class")


# ==============================================================================
# SECTION 8: SAVE RESULTS
# ==============================================================================


def save_results(metrics_df: pd.DataFrame) -> None:
    """
    Save metrics to CSV and Markdown formats.

    Args:
        metrics_df: DataFrame with metrics
    """
    print_section("6. SAVING RESULTS")

    # Save CSV
    metrics_df.to_csv(Config.OUTPUT_CSV, index=False, float_format='%.4f')
    print(f"✓ Metrics saved to: {Config.OUTPUT_CSV}")

    # Save Markdown
    with open(Config.OUTPUT_MD, 'w') as f:
        f.write("# Model Performance Comparison\n\n")
        f.write("## Evaluation Metrics (Test Set)\n\n")
        f.write(metrics_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n---\n\n")
        f.write("**Metrics Explanation:**\n\n")
        f.write(
            "- **Accuracy**: Overall correctness (can be misleading in imbalanced data)\n")
        f.write("- **Precision**: Of predicted 'in bed', how many were correct?\n")
        f.write("- **Recall**: Of actual 'in bed', how many did we detect?\n")
        f.write(
            "- **Specificity**: Of actual 'not in bed', how many did we correctly identify?\n")
        f.write(
            "- **F1-Score**: Harmonic mean of Precision and Recall (best overall metric)\n")
        f.write("- **ROC-AUC**: Area under ROC curve (discrimination ability)\n")

    print(f"✓ Formatted table saved to: {Config.OUTPUT_MD}")

    # Summary statistics
    print("\nMetrics Summary:")
    print(f"  Models evaluated: {len(metrics_df)}")
    # Exclude 'Model' column
    print(f"  Metrics calculated: {len(metrics_df.columns) - 1}")
    print(f"  Best F1-Score: {metrics_df['F1-Score'].max():.4f}")
    print(f"  Worst F1-Score: {metrics_df['F1-Score'].min():.4f}")


# ==============================================================================
# SECTION 9: FINAL SUMMARY
# ==============================================================================


def print_final_summary(metrics_df: pd.DataFrame) -> None:
    """
    Print final evaluation summary.

    Args:
        metrics_df: DataFrame with sorted metrics
    """
    print_section("EVALUATION SUMMARY")

    best_model = metrics_df.iloc[0]

    summary = f"""
✓ Model evaluation completed successfully!

MODELS EVALUATED: {len(metrics_df)}

BEST PERFORMING MODEL:
  🏆 {best_model['Model']}

  Key Metrics:
    - F1-Score:    {best_model['F1-Score']:.4f} ⭐
    - Accuracy:    {best_model['Accuracy']:.4f}
    - Precision:   {best_model['Precision']:.4f}
    - Recall:      {best_model['Recall']:.4f}
    - Specificity: {best_model['Specificity']:.4f}

OUTPUT FILES:
  - {Config.OUTPUT_CSV} (spreadsheet format)
  - {Config.OUTPUT_MD} (markdown table)

KEY TAKEAWAYS:
  1. F1-Score is the most reliable metric for this problem
  2. Don't rely solely on Accuracy in imbalanced datasets
  3. Consider Precision vs Recall trade-off for your use case
  4. Specificity ensures we don't over-predict "in bed"

NEXT STEPS:
  - Analyze confusion matrices (optional: create 05_confusion_matrices.py)
  - Plot ROC curves for visual comparison
  - Consider hyperparameter tuning for best model
  - Deploy best model for real-time predictions
  - Monitor performance on new data (model drift)

RECOMMENDATIONS:
  - Use {best_model['Model']} for deployment
  - Set decision threshold based on cost of FP vs FN
  - Implement monitoring for prediction distribution
  - Retrain periodically with new data
"""
    print(summary)


# ==============================================================================
# SECTION 10: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """
    Main function that orchestrates the entire evaluation pipeline.

    This function executes the complete evaluation workflow:
    1. Load ground truth labels
    2. Load all model predictions
    3. Calculate metrics for all models
    4. Create comparative table and identify best model
    5. Print detailed discussion and interpretation
    6. Save results
    7. Print final summary
    """
    print("=" * 80)
    print("MODEL EVALUATION - METRICS COMPARISON")
    print("=" * 80)

    # 1. Load ground truth labels
    y_train, y_test, baseline_accuracy = load_ground_truth()

    # 2. Load all model predictions
    predictions = load_all_predictions()

    # 3. Calculate metrics for all models
    metrics_df = calculate_all_metrics(predictions, y_test)

    # 4. Create comparative table and identify best model
    metrics_df_sorted = create_comparative_table(metrics_df, baseline_accuracy)

    # 5. Print detailed discussion and interpretation
    print_detailed_discussion(metrics_df_sorted)

    # 6. Save results
    save_results(metrics_df_sorted)

    # 7. Print final summary
    print_final_summary(metrics_df_sorted)

    print("=" * 80)
    print("EVALUATION COMPLETED!")
    print("=" * 80)


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
