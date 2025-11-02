#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
IN BED PREDICTION - CONFUSION MATRIX ANALYSIS
==============================================================================

Purpose: Analyze prediction errors in detail for the best model

This script:
1. Identifies best model from comparative metrics (by F1-Score)
2. Loads model predictions and ground truth labels
3. Calculates confusion matrix (TN, FP, FN, TP)
4. Computes derived metrics (FPR, FNR, Specificity, Recall)
5. Provides contextual interpretation of errors (impact on sleep quality)
6. Creates professional confusion matrix visualization
7. Saves analysis as PNG and PDF for reports

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
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')


# Matplotlib Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 11


# Configuration Constants
class Config:
    """Confusion matrix analysis configuration parameters."""
    PROCESSED_DATA_DIR = Path('outputs/data_processed')
    MODELS_DIR = Path('outputs/models')
    PREDICTIONS_DIR = Path('outputs/predictions')
    OUTPUT_DIR = Path('outputs')

    # Input/Output files
    METRICS_CSV = 'outputs/comparative_metrics.csv'
    OUTPUT_PNG = 'confusion_matrix.png'
    OUTPUT_PDF = 'confusion_matrix.pdf'

    # Visualization settings
    DPI = 300  # Resolution for PNG output
    ANNOTATION_FONTSIZE = 14
    LABEL_FONTSIZE = 13
    TITLE_FONTSIZE = 16
    SUBTITLE_FONTSIZE = 11


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
# SECTION 3: BEST MODEL SELECTION
# ==============================================================================


def select_best_model() -> Tuple[str, float, pd.Series]:
    """
    Load metrics CSV and identify best model by F1-Score.

    Returns:
        Tuple[str, float, pd.Series]: best_model_name, best_f1_score, metrics_row

    Raises:
        SystemExit: If metrics CSV not found
    """
    print_section("1. SELECTING BEST MODEL")

    # Load metrics
    if not Path(Config.METRICS_CSV).exists():
        print(f"✗ ERROR: {Config.METRICS_CSV} not found!")
        print("  Please run 04_evaluate_metrics.py first.")
        exit(1)

    metrics_df = pd.read_csv(Config.METRICS_CSV)
    print(f"✓ Loaded metrics from: {Config.METRICS_CSV}")
    print(f"  Models evaluated: {len(metrics_df)}")

    # Find best model by F1-Score
    best_idx = metrics_df['F1-Score'].idxmax()
    best_model_row = metrics_df.loc[best_idx]
    best_model_name = best_model_row['Model']
    best_f1_score = best_model_row['F1-Score']

    print("\n🏆 Best model identified:")
    print(f"  Model: {best_model_name}")
    print(f"  F1-Score: {best_f1_score:.4f}")
    print(f"  Accuracy: {best_model_row['Accuracy']:.4f}")
    print(f"  Precision: {best_model_row['Precision']:.4f}")
    print(f"  Recall: {best_model_row['Recall']:.4f}")

    return best_model_name, best_f1_score, best_model_row


# ==============================================================================
# SECTION 4: DATA LOADING
# ==============================================================================


def load_model_data(model_name: str) -> Tuple[Any, pd.Series, Any]:
    """
    Load trained model, predictions, and ground truth.

    Args:
        model_name: Name of the best model

    Returns:
        Tuple[Any, pd.Series, Any]: model, y_test, y_pred

    Raises:
        SystemExit: If any required files not found or shapes don't match
    """
    print_section("2. LOADING MODEL DATA")

    # Load trained model
    model_path = Config.MODELS_DIR / f'model_{model_name}.pkl'
    if not model_path.exists():
        print(f"✗ ERROR: Model file not found: {model_path}")
        exit(1)

    model = load_pickle(model_path)
    print(f"✓ Model loaded: {model_path.name}")

    # Load test predictions
    pred_path = Config.PREDICTIONS_DIR / f'predictions_{model_name}_test.pkl'
    if not pred_path.exists():
        print(f"✗ ERROR: Predictions file not found: {pred_path}")
        exit(1)

    y_pred = load_pickle(pred_path)
    print(f"✓ Predictions loaded: {pred_path.name}")
    print(f"  Shape: {y_pred.shape}")

    # Load ground truth
    y_test_path = Config.PROCESSED_DATA_DIR / 'y_test.pkl'
    if not y_test_path.exists():
        print(f"✗ ERROR: Ground truth not found: {y_test_path}")
        exit(1)

    y_test = load_pickle(y_test_path)
    print(f"✓ Ground truth loaded: {y_test_path.name}")
    print(f"  Shape: {y_test.shape}")

    # Verify shapes match
    if y_test.shape != y_pred.shape:
        print("✗ ERROR: Shape mismatch!")
        print(f"  y_test: {y_test.shape}, y_pred: {y_pred.shape}")
        exit(1)

    print("\n✓ All data loaded successfully")

    return model, y_test, y_pred


# ==============================================================================
# SECTION 5: CONFUSION MATRIX CALCULATION
# ==============================================================================


def calculate_confusion_matrix(
        y_test: pd.Series, y_pred: Any) -> Tuple[Any, int, int, int, int]:
    """
    Calculate confusion matrix and extract components.

    Args:
        y_test: True labels
        y_pred: Predicted labels

    Returns:
        Tuple[Any, int, int, int, int]: cm, tn, fp, fn, tp
    """
    print_section("3. CALCULATING CONFUSION MATRIX")

    """
    CONFUSION MATRIX STRUCTURE (Binary Classification):

                    Predicted
                    0       1
        Actual  0   TN      FP
                1   FN      TP

    Where:
    - TN (True Negative):  Correctly predicted as "not in bed"
    - FP (False Positive): Incorrectly predicted as "in bed" (Type I Error)
    - FN (False Negative): Incorrectly predicted as "not in bed" (Type II Error)
    - TP (True Positive):  Correctly predicted as "in bed"

    Total predictions = TN + FP + FN + TP
    """

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    # Extract components
    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp

    print("\nConfusion Matrix Components:")
    print(f"  True Negatives (TN):  {tn:4d} ({tn / total * 100:5.2f}%)")
    print(f"  False Positives (FP): {fp:4d} ({fp / total * 100:5.2f}%)")
    print(f"  False Negatives (FN): {fn:4d} ({fn / total * 100:5.2f}%)")
    print(f"  True Positives (TP):  {tp:4d} ({tp / total * 100:5.2f}%)")
    print(f"  {'─' * 50}")
    print(f"  Total:                {total:4d}")

    return cm, tn, fp, fn, tp


# ==============================================================================
# SECTION 6: DERIVED METRICS CALCULATION
# ==============================================================================


def calculate_derived_metrics(
        tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
    """
    Calculate derived metrics from confusion matrix.

    Args:
        tn: True Negatives
        fp: False Positives
        fn: False Negatives
        tp: True Positives

    Returns:
        Dict[str, float]: Dictionary with derived metrics
    """
    print_section("4. CALCULATING DERIVED METRICS")

    """
    DERIVED METRICS EXPLANATION:

    1. FALSE POSITIVE RATE (FPR) = FP / (FP + TN)
       - Proportion of actual negatives incorrectly classified as positive
       - Also called: Fall-out, Type I Error Rate
       - Range: 0 to 1 (lower is better)
       - Interpretation: "Of all 'not in bed' cases, what % were wrong?"

    2. FALSE NEGATIVE RATE (FNR) = FN / (FN + TP)
       - Proportion of actual positives incorrectly classified as negative
       - Also called: Miss Rate, Type II Error Rate
       - Range: 0 to 1 (lower is better)
       - Interpretation: "Of all 'in bed' cases, what % did we miss?"

    3. SPECIFICITY (True Negative Rate) = TN / (TN + FP) = 1 - FPR
       - Proportion of actual negatives correctly identified
       - Range: 0 to 1 (higher is better)
       - Interpretation: "How good are we at identifying 'not in bed'?"

    4. RECALL/SENSITIVITY (True Positive Rate) = TP / (TP + FN) = 1 - FNR
       - Proportion of actual positives correctly identified
       - Range: 0 to 1 (higher is better)
       - Interpretation: "How good are we at detecting 'in bed'?"
    """

    # Calculate metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    metrics = {
        'FPR': fpr,
        'FNR': fnr,
        'Specificity': specificity,
        'Recall': recall
    }

    print("Derived Metrics:")
    print(f"  False Positive Rate (FPR): {fpr:.4f}")
    print(f"  False Negative Rate (FNR): {fnr:.4f}")
    print(f"  Specificity (TNR):         {specificity:.4f}")
    print(f"  Recall/Sensitivity (TPR):  {recall:.4f}")

    # Verification
    print("\nVerification:")
    print(f"  Specificity + FPR = {specificity + fpr:.4f} (should be 1.0)")
    print(f"  Recall + FNR = {recall + fnr:.4f} (should be 1.0)")

    return metrics


# ==============================================================================
# SECTION 7: CONTEXTUAL INTERPRETATION
# ==============================================================================


def print_contextual_interpretation(
        tn: int,
        fp: int,
        fn: int,
        tp: int,
        fpr: float,
        fnr: float) -> None:
    """
    Print detailed interpretation of errors in the context of this problem.

    Args:
        tn: True Negatives
        fp: False Positives
        fn: False Negatives
        tp: True Positives
        fpr: False Positive Rate
        fnr: False Negative Rate
    """
    print_section("5. CONTEXTUAL INTERPRETATION")

    """
    UNDERSTANDING ERRORS IN THE "IN BED DETECTION" PROBLEM

    This is not just a theoretical exercise - these errors have real-world
    implications for home automation and quality of life!
    """

    print("\n📊 ERROR ANALYSIS:\n")

    # False Positives Analysis
    print(f"1. FALSE POSITIVES (FP = {fp}):")
    print("   Definition: Model says 'in bed' but you're NOT actually in bed")
    print(f"   Rate: {fpr:.2%} of all 'not in bed' cases\n")

    print("   Real-world scenarios:")
    print("   - You're watching TV in the bedroom (lights off, but awake)")
    print("   - You're reading in bed but haven't gone to sleep yet")
    print("   - You're getting ready for bed (bathroom routine just finished)\n")

    print("   Consequences of FP:")
    print("   ❌ Sleep mode activates too early")
    print("   ❌ Phone goes to 'Do Not Disturb' while you're still active")
    print("   ❌ Smart lights won't turn on when you need them")
    print("   ❌ Temperature adjusts for sleep when you're still awake")
    print("   ❌ Miss important calls/notifications\n")

    print("   Impact: MODERATE annoyance, easily corrected manually\n")

    # False Negatives Analysis
    print(f"2. FALSE NEGATIVES (FN = {fn}):")
    print("   Definition: Model says 'NOT in bed' but you ARE actually in bed")
    print(f"   Rate: {fnr:.2%} of all 'in bed' cases\n")

    print("   Real-world scenarios:")
    print("   - You went to bed but system didn't detect it")
    print("   - You're trying to sleep but lights/heating haven't adjusted")
    print("   - You're in bed but notifications keep coming\n")

    print("   Consequences of FN:")
    print("   ❌ Sleep mode doesn't activate")
    print("   ❌ Lights remain on or turn on when you don't want them")
    print("   ❌ Phone notifications disturb your sleep")
    print("   ❌ Temperature not optimized for sleep")
    print("   ❌ Sleep quality degraded by environmental factors")
    print("   ❌ Smart alarm might not work as expected\n")

    print("   Impact: MORE SERIOUS - directly affects sleep quality\n")

    # Comparative Analysis
    print("3. WHICH ERROR IS WORSE?\n")

    print("   In this specific application:")
    print("   → FALSE NEGATIVES (FN) are generally WORSE")
    print("   → Why? Missing sleep detection degrades sleep quality")
    print("   → FP can be manually corrected; FN disrupts sleep\n")

    print("   However, context matters:")
    print("   • Security monitoring: FN critical (miss intrusion)")
    print("   • Medical alerts: FN critical (miss emergency)")
    print("   • Comfort automation: FN more impactful than FP")
    print("   • Entertainment (movie mode): FP more annoying\n")

    # Decision based on current data
    if fn > fp:
        print("   ⚠️  CURRENT MODEL STATUS:")
        print(f"   More False Negatives ({fn}) than False Positives ({fp})")
        print("   → Model is CONSERVATIVE (misses some 'in bed' cases)")
        print("   → Consider: Lowering decision threshold to increase Recall")
        print("   → Trade-off: Will increase FP but decrease FN")
    elif fp > fn:
        print("   ⚠️  CURRENT MODEL STATUS:")
        print(f"   More False Positives ({fp}) than False Negatives ({fn})")
        print("   → Model is AGGRESSIVE (over-predicts 'in bed')")
        print("   → Consider: Raising decision threshold to increase Precision")
        print("   → Trade-off: Will decrease FP but increase FN")
    else:
        print("   ✓ CURRENT MODEL STATUS:")
        print(f"   Balanced errors: FP={fp}, FN={fn}")
        print("   → Model treats both error types equally")

    print("\n4. RECOMMENDATIONS:\n")

    print("   For SLEEP QUALITY optimization:")
    print("   ✓ Prioritize HIGH RECALL (minimize FN)")
    print("   ✓ Accept some FP as tolerable trade-off")
    print("   ✓ Threshold tuning: Consider 0.3-0.4 instead of 0.5")
    print("   ✓ Add manual override button for edge cases\n")

    print("   For GENERAL HOME AUTOMATION:")
    print("   ✓ Balance Precision and Recall (current F1-Score approach)")
    print("   ✓ Implement confidence thresholds for different actions")
    print("   ✓ High-confidence → Full automation")
    print("   ✓ Low-confidence → Suggest actions, wait for confirmation")


# ==============================================================================
# SECTION 8: VISUALIZATION
# ==============================================================================


def create_confusion_matrix_plot(
    cm: Any,
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    model_name: str,
    metrics_row: pd.Series
) -> None:
    """
    Create and save confusion matrix heatmap.

    Args:
        cm: Confusion matrix
        tn: True Negatives
        fp: False Positives
        fn: False Negatives
        tp: True Positives
        model_name: Name of the model
        metrics_row: Row from metrics DataFrame with scores
    """
    print_section("6. CREATING VISUALIZATION")

    # Create output directory
    Config.OUTPUT_DIR.mkdir(exist_ok=True)

    # Calculate percentages
    total = tn + fp + fn + tp
    tn_pct = tn / total * 100
    fp_pct = fp / total * 100
    fn_pct = fn / total * 100
    tp_pct = tp / total * 100

    # Create figure
    _, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='white',
                square=True, ax=ax)

    # Add custom annotations with counts and percentages
    annotations = [
        [f'TN\n{tn}\n({tn_pct:.1f}%)', f'FP\n{fp}\n({fp_pct:.1f}%)'],
        [f'FN\n{fn}\n({fn_pct:.1f}%)', f'TP\n{tp}\n({tp_pct:.1f}%)']
    ]

    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5, annotations[i][j],
                    ha='center', va='center',
                    fontsize=Config.ANNOTATION_FONTSIZE, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    # Labels
    ax.set_xlabel(
        'Predicted Label',
        fontsize=Config.LABEL_FONTSIZE,
        fontweight='bold')
    ax.set_ylabel(
        'True Label',
        fontsize=Config.LABEL_FONTSIZE,
        fontweight='bold')
    ax.set_xticklabels(['Not in Bed (0)', 'In Bed (1)'], fontsize=11)
    ax.set_yticklabels(['Not in Bed (0)', 'In Bed (1)'],
                       fontsize=11, rotation=0)

    # Title with metrics
    title = f'Confusion Matrix - {model_name}'
    subtitle = (f"Accuracy: {metrics_row['Accuracy']:.2%} | "
                f"F1-Score: {metrics_row['F1-Score']:.4f} | "
                f"ROC-AUC: {metrics_row['ROC-AUC']:.4f}")

    plt.title(title, fontsize=Config.TITLE_FONTSIZE, fontweight='bold', pad=20)
    plt.suptitle(subtitle, fontsize=Config.SUBTITLE_FONTSIZE, y=0.96)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save as PNG
    png_path = Config.OUTPUT_DIR / Config.OUTPUT_PNG
    plt.savefig(png_path, dpi=Config.DPI, bbox_inches='tight')
    print(f"✓ PNG saved: {png_path}")

    # Save as PDF (vector format)
    pdf_path = Config.OUTPUT_DIR / Config.OUTPUT_PDF
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ PDF saved: {pdf_path}")

    plt.close()

    print(f"\n✓ Visualizations saved to: {Config.OUTPUT_DIR.absolute()}")


# ==============================================================================
# SECTION 9: EXECUTIVE SUMMARY
# ==============================================================================


def print_executive_summary(
    model_name: str,
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    metrics: Dict[str, float]
) -> None:
    """
    Print executive summary of confusion matrix analysis.

    Args:
        model_name: Name of the model
        tn: True Negatives
        fp: False Positives
        fn: False Negatives
        tp: True Positives
        metrics: Dictionary with derived metrics
    """
    print_section("EXECUTIVE SUMMARY")

    total = tn + fp + fn + tp
    tn_pct = tn / total * 100
    fp_pct = fp / total * 100
    fn_pct = fn / total * 100
    tp_pct = tp / total * 100

    # Determine performance level
    accuracy = (tn + tp) / total
    if accuracy > 0.95:
        performance = 'EXCELLENT'
    elif accuracy > 0.90:
        performance = 'GOOD'
    elif accuracy > 0.80:
        performance = 'MODERATE'
    else:
        performance = 'NEEDS IMPROVEMENT'

    summary = f"""
MODEL: {model_name}

CONFUSION MATRIX BREAKDOWN:
────────────────────────────────────────────────────────────────
  True Negatives (TN):  {tn:4d} ({tn_pct:5.1f}%)
    → Correctly identified as NOT in bed
    → Model correctly detected when you were awake/active

  False Positives (FP): {fp:4d} ({fp_pct:5.1f}%)
    → Incorrectly identified as in bed
    → Model thought you were sleeping when you weren't
    → Impact: Premature activation of sleep mode

  False Negatives (FN): {fn:4d} ({fn_pct:5.1f}%)
    → Incorrectly identified as NOT in bed
    → Model missed that you were actually sleeping
    → Impact: Sleep mode not activated, poor sleep quality

  True Positives (TP):  {tp:4d} ({tp_pct:5.1f}%)
    → Correctly identified as in bed
    → Model correctly detected when you were sleeping
────────────────────────────────────────────────────────────────

ERROR RATES:
  False Positive Rate (FPR): {metrics['FPR']:.4f} ({metrics['FPR'] * 100:.2f}%)
    → Of all 'not in bed' moments, {metrics['FPR'] * 100:.2f}% wrongly flagged

  False Negative Rate (FNR): {metrics['FNR']:.4f} ({metrics['FNR'] * 100:.2f}%)
    → Of all 'in bed' moments, {metrics['FNR'] * 100:.2f}% were missed

DETECTION CAPABILITIES:
  Specificity:          {metrics['Specificity']:.4f} ({metrics['Specificity'] * 100:.2f}%)
    → Ability to correctly identify 'not in bed'

  Recall/Sensitivity:   {metrics['Recall']:.4f} ({metrics['Recall'] * 100:.2f}%)
    → Ability to correctly detect 'in bed'

KEY INSIGHTS:
  • Total predictions analyzed: {total:,}
  • Correct predictions: {tn + tp:,} ({(tn + tp) / total * 100:.1f}%)
  • Incorrect predictions: {fp + fn:,} ({(fp + fn) / total * 100:.1f}%)
  • Model performance: {performance}

RECOMMENDATIONS:
  1. Monitor FN rate closely (impacts sleep quality directly)
  2. Consider threshold tuning if FN > FP significantly
  3. Implement confidence-based automation levels
  4. Add manual override for edge cases
  5. Collect more data for minority class ('in bed')

OUTPUT FILES:
  - {Config.OUTPUT_DIR}/{Config.OUTPUT_PNG} (high-resolution image)
  - {Config.OUTPUT_DIR}/{Config.OUTPUT_PDF} (vector format for reports)
"""
    print(summary)


# ==============================================================================
# SECTION 10: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """
    Main function that orchestrates confusion matrix analysis.

    This function executes the complete analysis workflow:
    1. Select best model
    2. Load model data
    3. Calculate confusion matrix
    4. Calculate derived metrics
    5. Print contextual interpretation
    6. Create visualization
    7. Print executive summary
    """
    print("=" * 80)
    print("CONFUSION MATRIX ANALYSIS - BEST MODEL")
    print("=" * 80)

    # 1. Select best model
    model_name, f1_score, metrics_row = select_best_model()

    # 2. Load model data
    model, y_test, y_pred = load_model_data(model_name)

    # 3. Calculate confusion matrix
    cm, tn, fp, fn, tp = calculate_confusion_matrix(y_test, y_pred)

    # 4. Calculate derived metrics
    metrics = calculate_derived_metrics(tn, fp, fn, tp)

    # 5. Print contextual interpretation
    print_contextual_interpretation(
        tn, fp, fn, tp, metrics['FPR'], metrics['FNR'])

    # 6. Create visualization
    create_confusion_matrix_plot(cm, tn, fp, fn, tp, model_name, metrics_row)

    # 7. Print executive summary
    print_executive_summary(model_name, tn, fp, fn, tp, metrics)

    print("=" * 80)
    print("CONFUSION MATRIX ANALYSIS COMPLETED!")
    print("=" * 80)


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
