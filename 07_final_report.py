#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
IN BED PREDICTION - FINAL REPORT GENERATION
==============================================================================

Purpose: Generate comprehensive Markdown report summarizing entire project

This script:
1. Aggregates results from all previous analysis scripts
2. Compiles project introduction and methodology
3. Summarizes exploratory data analysis findings
4. Documents preprocessing steps and decisions
5. Presents model comparison and evaluation results
6. Includes confusion matrix and ROC curve insights
7. Provides conclusions and recommendations for deployment

Author: Bruno Silva
Date: 2025
==============================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ==============================================================================

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Any

warnings.filterwarnings('ignore')


# Get library versions
try:
    import sklearn
    SKLEARN_VERSION = sklearn.__version__
except Exception:
    SKLEARN_VERSION = "Unknown"

PANDAS_VERSION = pd.__version__
NUMPY_VERSION = np.__version__


# Configuration Constants
class Config:
    """Final report generation configuration parameters."""
    OUTPUT_DIR = Path('outputs')
    REPORT_FILE = Path(OUTPUT_DIR / 'FINAL_REPORT.md')
    DATA_PROCESSED_DIR = Path(OUTPUT_DIR / 'data_processed')

    # Input files
    DATASET_CSV = Path(OUTPUT_DIR / 'dataset.csv')
    TRAINING_TIMES_CSV = Path(OUTPUT_DIR / 'training_times.csv')
    COMPARATIVE_METRICS_CSV = Path(OUTPUT_DIR / 'comparative_metrics.csv')
    AUC_COMPARISON_CSV = Path(OUTPUT_DIR / 'auc_comparison.csv')
    METADATA_PKL = Path(OUTPUT_DIR / 'metadata.pkl')

    # Output images
    CONFUSION_MATRIX_PNG = OUTPUT_DIR / 'confusion_matrix.png'
    ROC_CURVES_PNG = OUTPUT_DIR / 'roc_curves.png'


# ==============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ==============================================================================


def print_progress(message: str) -> None:
    """
    Print progress message.

    Args:
        message: Progress message to display
    """
    print(f"  ✓ {message}")


def safe_read_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    Safely read CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        Optional[pd.DataFrame]: DataFrame if successful, None otherwise
    """
    try:
        if Path(filepath).exists():
            return pd.read_csv(filepath)
    except Exception:
        pass
    return None


def load_pickle_safe(filepath: Path) -> Optional[Any]:
    """
    Safely load pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Optional[Any]: Unpickled object if successful, None otherwise
    """
    try:
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


# ==============================================================================
# SECTION 3: REPORT HEADER
# ==============================================================================


def write_header() -> str:
    """
    Generate report header.

    Returns:
        str: Formatted header section
    """
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = f"""# Final Report - Classification: Predicting In Bed Probability

**Project:** 01.1 - Supervised Learning - Binary Classification
**Date:** {current_date}
**Author:** Bruno Silva

---

**Environment:**
- pandas: {PANDAS_VERSION}
- NumPy: {NUMPY_VERSION}
- scikit-learn: {SKLEARN_VERSION}

---

"""
    return content


# ==============================================================================
# SECTION 4: INTRODUCTION
# ==============================================================================


def write_introduction() -> str:
    """
    Generate introduction section.

    Returns:
        str: Formatted introduction section
    """
    content = """## 1. INTRODUCTION

### Project Objective
Develop a binary classification model to predict whether a person is lying in bed based on home automation sensor data.

### Business Problem
- **Home Automation**: Adjust lighting, temperature, notifications automatically
- **Sleep Monitoring**: Track sleep patterns and optimize environment
- **Energy Efficiency**: Reduce power consumption during sleep
- **Comfort**: Activate "sleep mode" at the right time

### Task Type
- **Supervised Binary Classification**
- **Target**: `in_bed` (0 = not in bed, 1 = in bed)
- **Challenge**: Class imbalance

### Data Sources
7 CSV files with sensor data:
1. Bedroom Blinds
2. Hallway Light
3. Bedroom Light
4. WC Light
5. Bedroom TV
6. Hallway Motion Sensor
7. Sleep Tracking (ground truth)

---

"""
    return content


# ==============================================================================
# SECTION 5: EXPLORATORY DATA ANALYSIS
# ==============================================================================


def write_eda() -> str:
    """
    Generate EDA section.

    Returns:
        str: Formatted EDA section
    """
    content = """## 2. EXPLORATORY DATA ANALYSIS

"""

    df = safe_read_csv(Config.DATASET_CSV)
    if df is not None and 'in_bed' in df.columns:
        counts = df['in_bed'].value_counts().sort_index()
        pct = df['in_bed'].value_counts(normalize=True).sort_index() * 100

        imbalance_ratio = counts.get(
            0, 0) / counts.get(1, 1) if counts.get(1, 0) > 0 else 0

        content += f"""### Dataset Summary
- **Total Records**: {len(df):,}
- **Features**: {len(df.columns)}

### Class Distribution

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| 0 | Not in Bed | {counts.get(0, 0):,} | {pct.get(0, 0):.2f}% |
| 1 | In Bed | {counts.get(1, 0):,} | {pct.get(1, 0):.2f}% |

**Imbalance Ratio**: {imbalance_ratio:.2f}:1

This imbalance is expected - people spend more time awake (~16h) than sleeping (~8h).

"""

    content += """### Key Insights
- Hour of day highly correlated with target
- Sequential patterns (motion → WC → lights off) indicate bedtime routine
- Engineered 30-minute window features capture routine patterns
- All lights off is strong indicator of sleep

---

"""
    return content


# ==============================================================================
# SECTION 6: PREPROCESSING
# ==============================================================================


def write_preprocessing() -> str:
    """
    Generate preprocessing section.

    Returns:
        str: Formatted preprocessing section
    """
    content = """## 3. PREPROCESSING PIPELINE

### Steps Executed

**1. Data Consolidation**
- Merged 7 CSV files by timestamp
- Removed 'unavailable' states
- Forward fill for last valid sensor state

**2. Target Variable Creation**
- Labeled timestamps within sleep windows as `in_bed=1`
- Used sleep tracking data as ground truth

**3. One-Hot Encoding**
- Applied to categorical sensor states
- Prevents false ordinal relationships

**4. Stratified Train-Test Split (80/20)**
- Maintains class proportions
- Critical for imbalanced data

**5. StandardScaler**
- Fitted ONLY on training data (prevents data leakage)
- Essential for KNN

"""

    # Add dimensions if metadata exists
    meta = load_pickle_safe(Config.METADATA_PKL)
    if meta is not None:
        content += f"""
### Final Dimensions

| Split | Samples | Features |
|-------|---------|----------|
| Train | {meta['train_shape'][0]:,} | {meta['n_features']} |
| Test | {meta['test_shape'][0]:,} | {meta['n_features']} |

"""

    content += "---\n\n"
    return content


# ==============================================================================
# SECTION 7: MODELS
# ==============================================================================


def write_models() -> str:
    """
    Generate models section.

    Returns:
        str: Formatted models section
    """
    content = """## 4. MODELS TRAINED

### Algorithms

1. **Logistic Regression** - Linear baseline, interpretable
2. **K-Nearest Neighbors** - Non-parametric, local patterns
3. **SVM Linear** - Optimal hyperplane, high dimensions
4. **SVM RBF** - Non-linear decision boundaries
5. **Naïve Bayes** - Fast, probabilistic

"""

    df_times = safe_read_csv(Config.TRAINING_TIMES_CSV)
    if df_times is not None:
        content += "### Training Times\n\n"
        content += df_times.to_markdown(index=False, floatfmt=".4f")
        content += "\n\n"

    content += "---\n\n"
    return content


# ==============================================================================
# SECTION 8: RESULTS
# ==============================================================================


def write_results() -> str:
    """
    Generate results section.

    Returns:
        str: Formatted results section
    """
    content = """## 5. RESULTS AND METRICS

"""

    df_metrics = safe_read_csv(Config.COMPARATIVE_METRICS_CSV)
    if df_metrics is not None:
        content += "### Comparative Metrics\n\n"
        content += df_metrics.to_markdown(index=False, floatfmt=".4f")
        content += "\n\n"

        best_model = df_metrics.loc[df_metrics['F1-Score'].idxmax(), 'Model']
        best_f1 = df_metrics.loc[df_metrics['F1-Score'].idxmax(), 'F1-Score']

        content += f"**Best Model**: {best_model} (F1-Score: {
            best_f1:.4f})\n\n"

    content += """### Why Accuracy Is Not Enough

In imbalanced data, a model that always predicts the majority class can have high accuracy but be useless.

**Key Metrics**:
- **Precision**: Of predicted "in bed", how many correct?
- **Recall**: Of actual "in bed", how many detected?
- **F1-Score**: Balance between Precision and Recall
- **ROC-AUC**: Discrimination ability across thresholds

---

"""
    return content


# ==============================================================================
# SECTION 9: CONFUSION MATRIX
# ==============================================================================


def write_confusion_matrix() -> str:
    """
    Generate confusion matrix section.

    Returns:
        str: Formatted confusion matrix section
    """
    content = """## 6. CONFUSION MATRIX

"""

    if Config.CONFUSION_MATRIX_PNG.exists():
        content += f"![Confusion Matrix]({
            Config.CONFUSION_MATRIX_PNG.name})\n\n"

    content += """### Error Analysis

**False Positives (FP)**: Predicted "in bed" when awake
- Impact: Sleep mode activates prematurely
- Severity: MODERATE (annoying but correctable)

**False Negatives (FN)**: Predicted "not in bed" when sleeping
- Impact: Sleep mode doesn't activate, disrupts sleep
- Severity: HIGH (affects sleep quality directly)

**Conclusion**: FN is worse in this application - missing sleep detection defeats the purpose.

---

"""
    return content


# ==============================================================================
# SECTION 10: ROC CURVES
# ==============================================================================


def write_roc_curves() -> str:
    """
    Generate ROC curves section.

    Returns:
        str: Formatted ROC curves section
    """
    content = """## 7. ROC CURVES

"""

    if Config.ROC_CURVES_PNG.exists():
        content += f"![ROC Curves]({Config.ROC_CURVES_PNG.name})\n\n"

    df_auc = safe_read_csv(Config.AUC_COMPARISON_CSV)
    if df_auc is not None:
        content += "### AUC Scores\n\n"
        content += df_auc.to_markdown(index=False, floatfmt=".4f")
        content += "\n\n"

    content += """### Interpretation
- **AUC = 1.0**: Perfect classifier
- **AUC >= 0.9**: Excellent
- **AUC >= 0.8**: Good
- **AUC = 0.5**: Random guessing

---

"""
    return content


# ==============================================================================
# SECTION 11: CONCLUSIONS
# ==============================================================================


def write_conclusions() -> str:
    """
    Generate conclusions section.

    Returns:
        str: Formatted conclusions section
    """
    content = """## 8. CONCLUSIONS AND RECOMMENDATIONS

### 8.1 Recommended Model

"""

    df_metrics = safe_read_csv(Config.COMPARATIVE_METRICS_CSV)
    if df_metrics is not None:
        best_model = df_metrics.loc[df_metrics['F1-Score'].idxmax(), 'Model']
        content += f"**Primary Recommendation**: {best_model}\n\n"

    content += """### 8.2 Handling Imbalance

Strategies:
- Use `class_weight='balanced'` in models
- Try SMOTE (oversampling)
- Consider undersampling majority class

### 8.3 Threshold Optimization

Default threshold (0.5) may not be optimal:
- Lower threshold → Higher Recall (fewer FN)
- Higher threshold → Higher Precision (fewer FP)
- Optimize based on cost of FN vs FP

### 8.4 Feature Engineering

Potential improvements:
- Time since last sensor change
- Rolling window statistics
- Cyclical encoding for hour (sin/cos)
- Interaction features

### 8.5 Hyperparameter Tuning

Use GridSearchCV with StratifiedKFold:
- Logistic Regression: C, penalty
- KNN: n_neighbors, weights
- SVM: C, gamma

### 8.6 Next Steps

1. Implement threshold tuning
2. Try class weights
3. Collect more "in bed" samples
4. Deploy best model
5. Monitor performance over time

---

"""
    return content


# ==============================================================================
# SECTION 12: REPORT GENERATION
# ==============================================================================


def generate_report() -> None:
    """
    Generate complete report.

    This function orchestrates the generation of all report sections
    and saves the final Markdown report.
    """
    print("\n" + "=" * 80)
    print("GENERATING FINAL REPORT")
    print("=" * 80 + "\n")

    report = ""

    report += write_header()
    print_progress("Header")

    report += write_introduction()
    print_progress("Introduction")

    report += write_eda()
    print_progress("EDA")

    report += write_preprocessing()
    print_progress("Preprocessing")

    report += write_models()
    print_progress("Models")

    report += write_results()
    print_progress("Results")

    report += write_confusion_matrix()
    print_progress("Confusion Matrix")

    report += write_roc_curves()
    print_progress("ROC Curves")

    report += write_conclusions()
    print_progress("Conclusions")

    # Save Markdown report
    with open(Config.REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n✓ Markdown report saved: {Config.REPORT_FILE}")
    print(f"  Words: {len(report.split()):,}")
    print(f"  Size: {len(report.encode('utf-8')) / 1024:.2f} KB")

    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETED!")
    print("=" * 80)


# ==============================================================================
# SECTION 13: MAIN FUNCTION
# ==============================================================================


def main() -> None:
    """
    Main function that orchestrates report generation.
    """
    generate_report()


# ==============================================================================
# EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
