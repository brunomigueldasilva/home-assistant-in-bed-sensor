#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: 06_roc_curves.py
Objective: Generate and compare ROC curves for all trained models
Author: Bruno Silva
Date: 2025
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import pandas as pd
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings('ignore')


# Matplotlib Configuration
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


# Configuration Constants
class Config:
    """ROC curves analysis configuration parameters."""
    PROCESSED_DATA_DIR = Path('data_processed')
    MODELS_DIR = Path('models')
    OUTPUT_DIR = Path('outputs')

    # Model names (must match training script)
    MODEL_NAMES = [
        'LogisticRegression',
        'KNN',
        'SVM_Linear',
        'SVM_RBF',
        'NaiveBayes'
    ]

    # Colors for each model
    COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#06A77D']

    # Output files
    OUTPUT_PNG = 'roc_curves.png'
    OUTPUT_PDF = 'roc_curves.pdf'
    OUTPUT_CSV = 'auc_comparison.csv'
    METRICS_CSV = 'comparative_metrics.csv'

    # Visualization settings
    DPI = 300
    LINEWIDTH = 2.5
    BASELINE_LINEWIDTH = 1.5
    ANNOTATION_FONTSIZE = 10
    LABEL_FONTSIZE = 13
    TITLE_FONTSIZE = 16
    LEGEND_FONTSIZE = 11

    # AUC thresholds
    EXCELLENT_AUC = 0.9
    GOOD_AUC = 0.8
    ACCEPTABLE_AUC = 0.7
    POOR_AUC = 0.5


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


def load_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load test features and labels.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X_test, y_test

    Raises:
        SystemExit: If test data files not found
    """
    print_section("1. LOADING TEST DATA")

    # Load X_test
    X_test_path = Config.PROCESSED_DATA_DIR / 'X_test.pkl'
    if not X_test_path.exists():
        print(f"✗ ERROR: {X_test_path} not found!")
        print("  Please run 02_preprocessing.py first.")
        exit(1)

    X_test = load_pickle(X_test_path)
    print(f"✓ X_test loaded: {X_test.shape}")

    # Load y_test
    y_test_path = Config.PROCESSED_DATA_DIR / 'y_test.pkl'
    if not y_test_path.exists():
        print(f"✗ ERROR: {y_test_path} not found!")
        exit(1)

    y_test = load_pickle(y_test_path)
    print(f"✓ y_test loaded: {y_test.shape}")

    # Show class distribution
    print("\nTest set class distribution:")
    counts = y_test.value_counts().sort_index()
    pct = y_test.value_counts(normalize=True).sort_index() * 100
    print(
        f"  Class 0 (not in bed): {
            counts.get(
                0, 0):,} ({
            pct.get(
                0, 0):.2f}%)")
    print(
        f"  Class 1 (in bed):     {
            counts.get(
                1, 0):,} ({
            pct.get(
                1, 0):.2f}%)")

    return X_test, y_test


def load_models() -> Dict[str, Any]:
    """
    Load all trained models.

    Returns:
        Dict[str, Any]: Dictionary mapping model names to model objects

    Raises:
        SystemExit: If models directory not found
    """
    print_section("2. LOADING TRAINED MODELS")

    if not Config.MODELS_DIR.exists():
        print(f"✗ ERROR: Models directory not found: {Config.MODELS_DIR}")
        print("  Please run 03_train_models.py first.")
        exit(1)

    models = {}

    print("Loading models:\n")

    for model_name in Config.MODEL_NAMES:
        model_path = Config.MODELS_DIR / f'model_{model_name}.pkl'

        if not model_path.exists():
            print(f"  ⚠️  Warning: {model_name} not found, skipping...")
            continue

        model = load_pickle(model_path)
        models[model_name] = model
        print(f"  ✓ {model_name:<25} Type: {type(model).__name__}")

    print(f"\n✓ Loaded {len(models)} models")

    return models


# ==============================================================================
# SECTION 4: PROBABILITY EXTRACTION
# ==============================================================================


def get_prediction_probabilities(
        model: Any,
        X_test: pd.DataFrame,
        model_name: str) -> Optional[Any]:
    """
    Extract prediction probabilities or scores from model.

    Different models provide different types of outputs:
    - Probabilistic models (Logistic Regression, Naive Bayes, KNN):
      → Use predict_proba() for calibrated probabilities

    - SVM models:
      → Use decision_function() for uncalibrated scores
      → These are distances to hyperplane, not probabilities
      → But still useful for ranking and ROC curve

    Args:
        model: Trained model
        X_test: Test features
        model_name: Name of model (for logging)

    Returns:
        Optional[Any]: Array of scores/probabilities for positive class,
            or None if extraction failed
    """

    """
    PROBABILITY vs DECISION FUNCTION:

    Probabilities (predict_proba):
    - Range: [0, 1]
    - Interpretable as confidence/likelihood
    - Calibrated (0.7 means ~70% chance of being positive)
    - Available in: Logistic Regression, Naive Bayes, KNN, Random Forest

    Decision Function (decision_function):
    - Range: (-∞, +∞)
    - Not calibrated (magnitudes don't represent probabilities)
    - But ordering is meaningful: higher = more likely positive
    - Useful for ranking and ROC curves
    - Available in: SVM, Linear models with decision_function

    For ROC curves, we only care about ranking, so both work!
    """

    try:
        # Try predict_proba first (preferred for probabilistic models)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)[:, 1]
            print(
                f"  {model_name}: Using predict_proba (calibrated probabilities)")
            print(
                f"    Range: [{
                    probabilities.min():.4f}, {
                    probabilities.max():.4f}]")
            return probabilities

        # Fall back to decision_function (for SVM)
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(X_test)
            print(
                f"  {model_name}: Using decision_function (uncalibrated scores)")
            print(f"    Range: [{scores.min():.4f}, {scores.max():.4f}]")
            return scores

        # Last resort: use binary predictions (suboptimal)
        else:
            predictions = model.predict(X_test)
            print(
                f"  ⚠️  {model_name}: Using binary predictions (not ideal for ROC)")
            return predictions.astype(float)

    except Exception as e:
        print(f"  ✗ Error getting probabilities for {model_name}: {e}")
        return None


# ==============================================================================
# SECTION 5: ROC CURVE CALCULATION
# ==============================================================================


def calculate_roc_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate ROC curves for all models.

    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with ROC data for each model
    """
    print_section("3. CALCULATING ROC CURVES")

    """
    WHAT IS AN ROC CURVE?

    ROC (Receiver Operating Characteristic) curve visualizes model performance
    across ALL possible classification thresholds.

    For each threshold value (e.g., 0.1, 0.2, ..., 0.9):
    - Calculate TPR (True Positive Rate = Recall = TP / (TP + FN))
    - Calculate FPR (False Positive Rate = FP / (FP + TN))
    - Plot point (FPR, TPR)

    KEY INSIGHTS:
    - Perfect classifier: Goes through point (0, 1) - 100% TPR, 0% FPR
    - Random classifier: Diagonal line from (0, 0) to (1, 1)
    - Worse than random: Below diagonal (something is wrong!)

    THRESHOLD INTERPRETATION:
    - High threshold (e.g., 0.9): Conservative predictions
      → Low FPR (few false alarms), Low TPR (miss many positives)
      → Point near (0, 0) on curve

    - Low threshold (e.g., 0.1): Aggressive predictions
      → High FPR (many false alarms), High TPR (catch most positives)
      → Point near (1, 1) on curve

    - Default threshold (0.5): Middle ground
      → Balanced FPR and TPR
      → Point somewhere in middle of curve
    """

    print("\nExtracting probabilities and calculating ROC curves...\n")

    roc_data = {}

    for model_name, model in models.items():
        print(f"Processing {model_name}...")

        # Get probabilities/scores
        y_scores = get_prediction_probabilities(model, X_test, model_name)

        if y_scores is None:
            print(f"  ✗ Skipping {model_name} (failed to get probabilities)")
            continue

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)

        # Calculate AUC
        try:
            auc_score = roc_auc_score(y_test, y_scores)
        except Exception as e:
            print(f"  ✗ Error calculating AUC for {model_name}: {e}")
            continue

        # Store data
        roc_data[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc_score,
            'y_scores': y_scores
        }

        print(f"  ✓ ROC calculated: AUC = {auc_score:.4f}")
        print(f"    Number of thresholds: {len(thresholds)}")
        print()

    print(f"✓ ROC curves calculated for {len(roc_data)} models")

    return roc_data


# ==============================================================================
# SECTION 6: VISUALIZATION
# ==============================================================================


def plot_roc_curves(roc_data: Dict[str, Dict[str, Any]]) -> None:
    """
    Create comparative ROC curve visualization.

    Args:
        roc_data: Dictionary with ROC data for each model
    """
    print_section("4. CREATING ROC CURVE VISUALIZATION")

    """
    HOW TO INTERPRET AUC (Area Under Curve)?

    AUC summarizes the entire ROC curve into a single number:

    - AUC = 1.0: Perfect classifier
      → Always ranks positive samples higher than negative samples
      → ROC curve goes through (0, 1) corner

    - AUC = 0.9-1.0: Excellent discrimination
      → Model almost always ranks correctly

    - AUC = 0.8-0.9: Good discrimination
      → Model usually ranks correctly, some errors

    - AUC = 0.7-0.8: Acceptable discrimination
      → Model is better than random, but not great

    - AUC = 0.5-0.7: Poor discrimination
      → Barely better than random guessing

    - AUC = 0.5: Random classifier
      → No discriminative power (coin flip)

    - AUC < 0.5: Worse than random
      → Predictions are inverted! Check your labels or code!

    PROBABILISTIC INTERPRETATION:
    AUC = Probability that model ranks a random positive example
    higher than a random negative example.

    Example: AUC = 0.85 means 85% chance that a random "in bed" case
    will have higher predicted probability than a random "not in bed" case.
    """

    # Create figure
    _, ax = plt.subplots(figsize=(12, 8))

    # Plot each model's ROC curve
    for i, (model_name, data) in enumerate(roc_data.items()):
        ax.plot(data['fpr'], data['tpr'],
                color=Config.COLORS[i % len(Config.COLORS)],
                linewidth=Config.LINEWIDTH,
                label=f"{model_name} (AUC = {data['auc']:.4f})")

    # Plot random classifier baseline (diagonal)
    ax.plot([0, 1], [0, 1],
            'k--',
            linewidth=Config.BASELINE_LINEWIDTH,
            label='Random Classifier (AUC = 0.5000)',
            alpha=0.7)

    # Add annotations
    ax.text(0.05, 0.95,
            'Perfect Classifier\n(TPR=1, FPR=0)',
            transform=ax.transAxes,
            fontsize=Config.ANNOTATION_FONTSIZE,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    ax.text(0.95, 0.05,
            'Worse Performance\n(High FPR, Low TPR)',
            transform=ax.transAxes,
            fontsize=Config.ANNOTATION_FONTSIZE,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    # Labels and title
    ax.set_xlabel(
        'False Positive Rate (FPR)',
        fontsize=Config.LABEL_FONTSIZE,
        fontweight='bold')
    ax.set_ylabel(
        'True Positive Rate (TPR / Recall)',
        fontsize=Config.LABEL_FONTSIZE,
        fontweight='bold')
    ax.set_title('ROC Curves - Comparative Analysis of All Models',
                 fontsize=Config.TITLE_FONTSIZE, fontweight='bold', pad=20)

    # Legend
    ax.legend(
        loc='lower right',
        fontsize=Config.LEGEND_FONTSIZE,
        framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    # Add diagonal reference lines at 0.25, 0.5, 0.75
    for val in [0.25, 0.5, 0.75]:
        ax.axhline(
            y=val,
            color='gray',
            linestyle=':',
            alpha=0.2,
            linewidth=0.5)
        ax.axvline(
            x=val,
            color='gray',
            linestyle=':',
            alpha=0.2,
            linewidth=0.5)

    plt.tight_layout()

    # Save figures
    Config.OUTPUT_DIR.mkdir(exist_ok=True)

    png_path = Config.OUTPUT_DIR / Config.OUTPUT_PNG
    plt.savefig(png_path, dpi=Config.DPI, bbox_inches='tight')
    print(f"✓ PNG saved: {png_path}")

    pdf_path = Config.OUTPUT_DIR / Config.OUTPUT_PDF
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ PDF saved: {pdf_path}")

    plt.close()

    print(f"\n✓ Visualizations saved to: {Config.OUTPUT_DIR.absolute()}")


# ==============================================================================
# SECTION 7: AUC COMPARISON TABLE
# ==============================================================================


def create_auc_comparison_table(
        roc_data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create and save AUC comparison table.

    Args:
        roc_data: Dictionary with ROC data for each model

    Returns:
        pd.DataFrame: DataFrame with AUC comparison
    """
    print_section("5. AUC COMPARISON TABLE")

    """
    ROC-AUC vs F1-SCORE: WHEN DO THEY AGREE?

    Both metrics evaluate model quality but focus on different aspects:

    ROC-AUC:
    - Measures ranking ability across ALL thresholds
    - Insensitive to class imbalance
    - Evaluates discrimination capability
    - Good for: Understanding overall model quality

    F1-Score:
    - Evaluates performance at ONE specific threshold (usually 0.5)
    - Sensitive to class imbalance (focuses on positive class)
    - Balances Precision and Recall
    - Good for: Real-world deployment with fixed threshold

    THEY USUALLY AGREE:
    - Model with high AUC typically has high F1-Score

    THEY MAY DISAGREE WHEN:
    1. Default threshold (0.5) is not optimal
       → High AUC but moderate F1-Score
       → Solution: Tune threshold

    2. Class imbalance is extreme
       → High AUC but low F1-Score for minority class
       → Solution: Consider Precision-Recall curve instead

    3. Model has good ranking but poor calibration
       → High AUC but poor probabilities
       → Solution: Calibrate model probabilities
    """

    # Create comparison table
    comparison_data = []

    for model_name, data in roc_data.items():
        comparison_data.append({
            'Model': model_name,
            'AUC': data['auc']
        })

    # Create DataFrame
    auc_df = pd.DataFrame(comparison_data)

    # Sort by AUC descending
    auc_df = auc_df.sort_values('AUC', ascending=False).reset_index(drop=True)

    # Add ranking
    auc_df['Rank'] = range(1, len(auc_df) + 1)

    # Reorder columns
    auc_df = auc_df[['Rank', 'Model', 'AUC']]

    print("\nAUC Ranking:\n")
    print(auc_df.to_string(index=False))

    # Performance categories
    print("\n" + "─" * 80)
    print("Performance Categories:")
    for _, row in auc_df.iterrows():
        auc = row['AUC']
        if auc >= Config.EXCELLENT_AUC:
            category = "EXCELLENT"
            emoji = "🌟"
        elif auc >= Config.GOOD_AUC:
            category = "GOOD"
            emoji = "✓"
        elif auc >= Config.ACCEPTABLE_AUC:
            category = "ACCEPTABLE"
            emoji = "○"
        elif auc >= Config.POOR_AUC:
            category = "POOR"
            emoji = "⚠️"
        else:
            category = "WORSE THAN RANDOM"
            emoji = "✗"

        print(f"  {emoji} {row['Model']:<25} AUC: {auc:.4f} - {category}")

    # Save CSV
    auc_df.to_csv(Config.OUTPUT_CSV, index=False, float_format='%.4f')
    print(f"\n✓ AUC comparison saved: {Config.OUTPUT_CSV}")

    return auc_df


# ==============================================================================
# SECTION 8: CROSS-METRIC ANALYSIS
# ==============================================================================


def analyze_metric_agreement(auc_df: pd.DataFrame) -> None:
    """
    Compare AUC ranking with F1-Score ranking.

    Args:
        auc_df: DataFrame with AUC rankings
    """
    print_section("6. CROSS-METRIC ANALYSIS")

    # Try to load F1-Score comparison
    try:
        f1_df = pd.read_csv(Config.METRICS_CSV)
        f1_df = f1_df.sort_values(
            'F1-Score',
            ascending=False).reset_index(
            drop=True)

        print("Comparing rankings: AUC vs F1-Score\n")

        # Create comparison table
        print(
            f"{
                'Rank':<6} {
                'Model (by AUC)':<25} {
                'AUC':<8} {
                    'Model (by F1)':<25} {
                        'F1-Score':<10}")
        print("─" * 80)

        max_rank = max(len(auc_df), len(f1_df))

        for i in range(max_rank):
            auc_model = auc_df.iloc[i]['Model'] if i < len(auc_df) else "N/A"
            auc_score = auc_df.iloc[i]['AUC'] if i < len(auc_df) else 0

            f1_model = f1_df.iloc[i]['Model'] if i < len(f1_df) else "N/A"
            f1_score = f1_df.iloc[i]['F1-Score'] if i < len(f1_df) else 0

            # Check if rankings agree
            agreement = "✓" if auc_model == f1_model else "✗"

            print(
                f"{
                    i +
                    1:<6} {
                    auc_model:<25} {
                    auc_score:<8.4f} {
                    f1_model:<25} {
                        f1_score:<10.4f} {agreement}")

        # Identify top models
        best_auc_model = auc_df.iloc[0]['Model']
        best_f1_model = f1_df.iloc[0]['Model']

        print("\n" + "─" * 80)
        print("TOP PERFORMERS:")
        print(
            f"  Best by AUC:      {best_auc_model} (AUC = {
                auc_df.iloc[0]['AUC']:.4f})")
        print(
            f"  Best by F1-Score: {best_f1_model} (F1 = {f1_df.iloc[0]['F1-Score']:.4f})")

        if best_auc_model != best_f1_model:
            print("\n⚠️  ATTENTION: Rankings differ!")
            print("\n  Possible reasons:")
            print(f"  1. {best_auc_model} has good ranking ability (high AUC)")
            print("     but default threshold (0.5) is not optimal (lower F1)")
            print(f"  2. {best_f1_model} performs well at threshold 0.5")
            print("     but ranking ability is slightly lower")
            print("\n  Recommendation:")
            print(
                f"  - For threshold-independent evaluation: Use {best_auc_model} (AUC)")
            print(
                f"  - For deployment with fixed threshold: Use {best_f1_model} (F1)")
            print(f"  - Consider threshold tuning for {best_auc_model}")
        else:
            print("\n✓ Rankings AGREE: Both metrics recommend the same model!")
            print(f"  → {best_auc_model} is the clear winner")

    except FileNotFoundError:
        print(f"⚠️  {Config.METRICS_CSV} not found")
        print("  Run 04_evaluate_metrics.py to enable cross-metric comparison")


# ==============================================================================
# SECTION 9: FINAL SUMMARY
# ==============================================================================


def print_final_summary(
        roc_data: Dict[str, Dict[str, Any]], auc_df: pd.DataFrame) -> None:
    """
    Print final ROC analysis summary.

    Args:
        roc_data: Dictionary with ROC data for each model
        auc_df: DataFrame with AUC rankings
    """
    print_section("ROC ANALYSIS SUMMARY")

    best_model = auc_df.iloc[0]
    worst_model = auc_df.iloc[-1]

    # Determine performance categories
    best_performance = (
        "EXCELLENT" if best_model['AUC'] >= Config.EXCELLENT_AUC else
        "GOOD" if best_model['AUC'] >= Config.GOOD_AUC else
        "ACCEPTABLE"
    )

    worst_performance = (
        "EXCELLENT" if worst_model['AUC'] >= Config.EXCELLENT_AUC else
        "GOOD" if worst_model['AUC'] >= Config.GOOD_AUC else
        "ACCEPTABLE" if worst_model['AUC'] >= Config.ACCEPTABLE_AUC else
        "POOR"
    )

    summary = f"""
✓ ROC curve analysis completed successfully!

MODELS ANALYZED: {len(roc_data)}

BEST MODEL (by AUC):
  🏆 {best_model['Model']}
  AUC: {best_model['AUC']:.4f}
  Performance: {best_performance}

WORST MODEL (by AUC):
  {worst_model['Model']}
  AUC: {worst_model['AUC']:.4f}
  Performance: {worst_performance}

OUTPUT FILES:
  - {Config.OUTPUT_DIR}/{Config.OUTPUT_PNG} (high-resolution comparison plot)
  - {Config.OUTPUT_DIR}/{Config.OUTPUT_PDF} (vector format for publications)
  - {Config.OUTPUT_CSV} (AUC rankings table)

KEY INSIGHTS:
  • ROC curves show model performance across ALL thresholds
  • AUC summarizes discrimination ability in one number
  • Higher AUC = Better ranking of positive vs negative samples
  • AUC >= {Config.EXCELLENT_AUC} is considered excellent performance
  • Compare with F1-Score to understand threshold sensitivity

WHEN TO USE ROC-AUC:
  ✓ Evaluating model's inherent discrimination ability
  ✓ Comparing models independent of threshold choice
  ✓ Understanding trade-off between TPR and FPR
  ✓ When class balance is not extremely skewed

WHEN TO USE OTHER METRICS:
  • Extreme class imbalance → Precision-Recall curve
  • Fixed deployment threshold → F1-Score at that threshold
  • Cost-sensitive decisions → Custom cost matrix analysis

NEXT STEPS:
  1. If best AUC ≠ best F1: Consider threshold tuning
  2. Analyze Precision-Recall curves for minority class focus
  3. Investigate why some models have lower AUC
  4. Deploy best model with appropriate threshold
  5. Monitor AUC on new data to detect model drift
"""
    print(summary)


# ==============================================================================
# SECTION 10: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """
    Main function that orchestrates ROC curve analysis.

    This function executes the complete ROC analysis workflow:
    1. Load test data
    2. Load trained models
    3. Calculate ROC curves for all models
    4. Plot comparative ROC curves
    5. Create AUC comparison table
    6. Analyze agreement with F1-Score
    7. Print final summary
    """
    print("=" * 80)
    print("ROC CURVE ANALYSIS - ALL MODELS")
    print("=" * 80)

    # 1. Load test data
    X_test, y_test = load_test_data()

    # 2. Load trained models
    models = load_models()

    # 3. Calculate ROC curves for all models
    roc_data = calculate_roc_curves(models, X_test, y_test)

    # 4. Plot comparative ROC curves
    plot_roc_curves(roc_data)

    # 5. Create AUC comparison table
    auc_df = create_auc_comparison_table(roc_data)

    # 6. Analyze agreement with F1-Score
    analyze_metric_agreement(auc_df)

    # 7. Print final summary
    print_final_summary(roc_data, auc_df)

    print("=" * 80)
    print("ROC ANALYSIS COMPLETED!")
    print("=" * 80)


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
