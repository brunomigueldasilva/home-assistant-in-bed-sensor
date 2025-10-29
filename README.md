# Home Assistant - In Bed Sensor

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Machine Learning - Supervised Learning (Classification Models)
**Machine Learning pipeline for predicting whether a person is in bed using Home Assistant sensor data.**

This project demonstrates a complete supervised learning workflow for binary classification, transforming smart home automation data into actionable predictions that can optimize sleep environments, energy efficiency, and home automation routines.

---

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Architecture](#pipeline-architecture)
- [Results](#results)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

---

## Description

This project implements a **binary classification system** to predict whether a person is lying in bed based on sensor data collected from a Home Assistant smart home installation. The system analyzes patterns from multiple sensors including lights, motion detectors, blinds, and TV usage to make accurate predictions.

The complete machine learning pipeline includes:
- **Data collection** from 7 different CSV sensor files
- **Exploratory data analysis** with comprehensive visualizations
- **Feature engineering** including temporal patterns and 30-minute window features
- **Training of 5 different models** (Logistic Regression, KNN, SVM Linear, SVM RBF, Naïve Bayes)
- **Comprehensive evaluation** with multiple metrics
- **Automated reporting** with confusion matrices and ROC curves

### Why This Project?

Smart homes generate vast amounts of sensor data, but extracting meaningful insights requires sophisticated analysis. This project demonstrates how machine learning can:
- **Automate home environments** by detecting sleep patterns
- **Improve energy efficiency** by activating sleep modes at the right time
- **Enhance comfort** by adjusting lighting and temperature automatically
- **Monitor sleep quality** without intrusive wearable devices

---

## Features

### Core Capabilities
- ✅ **Automated ML Pipeline**: 7-step workflow from data loading to final report
- ✅ **Multiple Models**: Comparison of 5 different classification algorithms
- ✅ **Imbalanced Data Handling**: Stratified splitting and class-aware metrics
- ✅ **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- ✅ **Visual Analytics**: Confusion matrices, ROC curves, and exploratory plots
- ✅ **Production-Ready**: Includes orchestrator for automated execution
- ✅ **Detailed Reporting**: Auto-generated Markdown reports with insights

### Technical Features
- **Data Preprocessing**: Handles missing values, removes invalid states, forward-fills sensor data
- **Feature Engineering**: One-hot encoding for categorical variables, temporal features
- **Model Serialization**: Saves trained models and scalers using pickle
- **Stratified Splitting**: Maintains class proportions in train/test sets
- **Standardization**: Prevents data leakage by fitting only on training data
- **Reproducibility**: Seeded random states for consistent results

---

## Business Problem

### Challenge
How can we automatically detect when a person is in bed to optimize home automation systems?

### Impact Areas

| Area | Benefit | Example |
|------|---------|---------|
| **Home Automation** | Automatic mode switching | Activate "sleep mode" when in bed |
| **Energy Efficiency** | Reduce unnecessary power consumption | Turn off lights, lower heating/AC |
| **Sleep Monitoring** | Track sleep patterns without wearables | Optimize bedroom environment |
| **Comfort** | Personalized environment | Adjust lighting, temperature, notifications |
| **Safety** | Detect unusual patterns | Alert if expected bedtime missed |

### Target Audience
- Smart home enthusiasts
- Home automation developers
- Sleep researchers
- Energy efficiency consultants
- IoT/ML practitioners

---

## Dataset

### Data Sources

The project uses **7 CSV files** containing timestamped sensor data:

1. **bedroom_blinds.csv** - Bedroom blinds state
2. **hallway_light.csv** - Hallway light state
3. **bedoroom_light.csv** - Bedroom light state
4. **wc_light.csv** - Bathroom light state
5. **bedroom_tv.csv** - Bedroom TV state
6. **hallway_motion_sensor.csv** - Hallway motion sensor
7. **sleep.csv** - Ground truth sleep tracking data

### Data Characteristics

- **Temporal Data**: All sensors include `last_changed` timestamp
- **Categorical States**: Binary or multi-class states (on/off, open/closed, etc.)
- **Imbalanced Classes**: More "not in bed" samples than "in bed" (expected: ~2:1 ratio)
- **Sequential Patterns**: Captures bedtime routines (motion → WC → lights off)

### Target Variable

- **Variable**: `in_bed`
- **Type**: Binary (0 = Not in Bed, 1 = In Bed)
- **Labels Created**: Using sleep tracking data as ground truth

### Example Data Structure

```csv
last_changed,state,sensor,in_bed
2024-01-15 22:30:15,off,bedroom_light,0
2024-01-15 23:15:42,on,motion_sensor,0
2024-01-15 23:45:20,off,tv,1
2024-01-16 00:10:05,off,hallway_light,1
```

---

## Installation

### Requirements

- **Python**: 3.13 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: ~500 MB - 2 GB (depends on dataset size)
- **Storage**: ~10MB for dependencies + your dataset size

### Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

**Core Libraries:**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

**Optional (for enhanced output):**
```
colorama>=0.4.4  # Colored terminal output
```

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/brunomigueldasilva/home-assistant-in-bed-sensor.git
cd home-assistant-in-bed-sensor
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Prepare your data**:
```bash
mkdir inputs
# Place your CSV files in the inputs/ folder
```

### Directory Structure

```
home-assistant-in-bed-sensor/
│
├── inputs/                      # Place your CSV files here
│   ├── bedroom_blinds.csv
│   ├── hallway_light.csv
│   ├── bedoroom_light.csv
│   ├── wc_light.csv
│   ├── bedroom_tv.csv
│   ├── hallway_motion_sensor.csv
│   └── sleep.csv
│
├── 01_exploratory_analysis.py   # Step 1: EDA
├── 02_preprocessing.py          # Step 2: Data preprocessing
├── 03_train_models.py           # Step 3: Model training
├── 04_evaluate_metrics.py       # Step 4: Metrics evaluation
├── 05_confusion_matrix.py       # Step 5: Confusion matrix
├── 06_roc_curves.py             # Step 6: ROC curves
├── 07_final_report.py           # Step 7: Final report
├── 08_orchestrator.py           # Pipeline orchestrator
│
├── outputs/                     # Generated plots (auto-created)
├── data_processed/              # Processed datasets (auto-created)
├── models/                      # Trained models (auto-created)
├── predictions/                 # Model predictions (auto-created)
│
├── requirements.txt
├── README.md
├── QUICK_START.md
├── CONTRIBUTING.md
├── LICENSE
└── PROJECT_OVERVIEW
```

---

## Usage

### Quick Start

Run the complete pipeline with one command:

```bash
python 08_orchestrator.py --all
```

### Interactive Mode

Launch the interactive menu:

```bash
python 08_orchestrator.py
```

**Menu Options:**
1. Run complete pipeline (all 7 scripts)
2. Run specific steps (choose which ones)
3. Resume execution (from last failed step)
4. Clean outputs and restart
5. Exit

### Run Specific Steps

Execute only certain steps:

```bash
python 08_orchestrator.py --steps 1,3,7
```

### Individual Script Execution

You can also run scripts individually:

```bash
python 01_exploratory_analysis.py
python 02_preprocessing.py
python 03_train_models.py
# ... and so on
```

### Command Line Options

```bash
# Run complete pipeline (non-interactive)
python 08_orchestrator.py --all

# Run specific steps
python 08_orchestrator.py --steps 1,3,4

# Clean all outputs
python 08_orchestrator.py --clean

# Silent mode (no user interaction)
python 08_orchestrator.py --all --silent
```

---

## Pipeline Architecture

The project follows a **7-step automated machine learning pipeline**:

### Step 1: Exploratory Data Analysis
**Script**: `01_exploratory_analysis.py`

- Loads and validates 7 CSV sensor files
- Removes invalid states ('unavailable', 'unknown')
- Merges all sensors by timestamp
- Creates target variable using sleep tracking data
- Generates comprehensive visualizations
- Analyzes class distribution and feature correlations

**Outputs**:
- `dataset.csv` - Consolidated dataset
- `outputs/` - EDA visualizations

---

### Step 2: Data Preprocessing
**Script**: `02_preprocessing.py`

- Applies one-hot encoding to categorical variables
- Creates train-test split (80/20) with stratification
- Fits StandardScaler on training data only (prevents data leakage)
- Saves processed data and preprocessing objects

**Outputs**:
- `data_processed/X_train.pkl` - Training features
- `data_processed/X_test.pkl` - Test features
- `data_processed/y_train.pkl` - Training labels
- `data_processed/y_test.pkl` - Test labels
- `data_processed/scaler.pkl` - Fitted scaler
- `data_processed/metadata.pkl` - Dataset metadata

---

### Step 3: Model Training
**Script**: `03_train_models.py`

Trains **5 classification models**:
1. **Logistic Regression** - Linear baseline model
2. **K-Nearest Neighbors (KNN)** - Non-parametric classifier
3. **SVM Linear** - Support Vector Machine with linear kernel
4. **SVM RBF** - SVM with radial basis function kernel
5. **Naïve Bayes** - Probabilistic classifier

**Outputs**:
- `models/*.pkl` - Serialized trained models
- `predictions/*.csv` - Model predictions
- `training_times.csv` - Training duration for each model

---

### Step 4: Metrics Evaluation
**Script**: `04_evaluate_metrics.py`

Calculates comprehensive metrics:
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

**Outputs**:
- `comparative_metrics.csv` - All metrics comparison
- `comparative_metrics.md` - Formatted table

---

### Step 5: Confusion Matrix
**Script**: `05_confusion_matrix.py`

- Generates confusion matrices for all models
- Visualizes True Positives, False Positives, True Negatives, False Negatives
- Provides error analysis

**Outputs**:
- `outputs/confusion_matrix.png` - Confusion matrix visualization

---

### Step 6: ROC Curves
**Script**: `06_roc_curves.py`

- Plots ROC curves for all models
- Calculates AUC scores
- Compares model discrimination ability

**Outputs**:
- `outputs/roc_curves.png` - ROC curves comparison
- `auc_comparison.csv` - AUC scores table

---

### Step 7: Final Report
**Script**: `07_final_report.py`

- Generates comprehensive Markdown report
- Includes all metrics, visualizations, and insights
- Provides recommendations and next steps

**Outputs**:
- `FINAL_REPORT.md` - Complete analysis report

---

## Results

### Performance Overview

After training 5 different models, the system achieves strong performance in detecting sleep patterns:

| Metric | Best Model Range |
|--------|------------------|
| **Accuracy** | 85-95% |
| **Precision** | 80-92% |
| **Recall** | 78-90% |
| **F1-Score** | 80-91% |
| **ROC-AUC** | 0.90-0.98 |

*Note: Exact values depend on your specific dataset.*

### Key Insights

1. **Time-based features are highly predictive**: Hour of day strongly correlates with in-bed status
2. **Sequential patterns matter**: Motion → WC → lights off indicates bedtime routine
3. **All lights off is a strong signal**: Excellent predictor of sleep state
4. **Imbalanced data is expected**: People spend ~16h awake vs ~8h sleeping (2:1 ratio)

### Error Analysis

- **False Positives (FP)**: Predicted "in bed" when actually awake
  - Impact: MODERATE - Sleep mode activates prematurely (annoying but correctable)
  
- **False Negatives (FN)**: Predicted "not in bed" when actually sleeping
  - Impact: HIGH - Sleep mode doesn't activate, disrupts sleep quality

**Recommendation**: In this application, minimizing False Negatives is more critical than minimizing False Positives.

---

## Model Performance

### Model Comparison

Each model has unique characteristics:

#### 1. Logistic Regression
- **Pros**: Fast, interpretable, handles high dimensions well
- **Cons**: Assumes linear relationships
- **Best for**: Baseline comparison, interpretability needed

#### 2. K-Nearest Neighbors (KNN)
- **Pros**: Non-parametric, captures local patterns
- **Cons**: Slower prediction, sensitive to scaling
- **Best for**: When local neighborhoods are important

#### 3. SVM Linear
- **Pros**: Effective in high dimensions, optimal hyperplane
- **Cons**: Sensitive to outliers
- **Best for**: Linearly separable data

#### 4. SVM RBF
- **Pros**: Handles non-linear boundaries, flexible
- **Cons**: Computationally expensive, requires tuning
- **Best for**: Complex decision boundaries

#### 5. Naïve Bayes
- **Pros**: Very fast, works well with small data
- **Cons**: Independence assumption may not hold
- **Best for**: Quick baseline, probabilistic predictions

### Recommendations

**For Production Deployment:**
- Use the model with highest F1-Score
- Consider ensemble methods combining multiple models
- Implement threshold tuning to optimize Recall (minimize FN)

**For Interpretability:**
- Use Logistic Regression with feature importance analysis

**For Best Performance:**
- Fine-tune SVM RBF or test ensemble methods
- Consider XGBoost or Random Forest (future work)

---

## Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Make your changes**
4. **Run tests** (if available)
5. **Commit your changes**:
   ```bash
   git commit -m "Add YourFeature"
   ```
6. **Push to the branch**:
   ```bash
   git push origin feature/YourFeature
   ```
7. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to all functions
- Include comments for complex logic
- Update README.md if adding new features
- Test your changes thoroughly

### Areas for Contribution

- **Feature Engineering**: Add new temporal or interaction features
- **Model Improvements**: Implement ensemble methods or deep learning
- **Hyperparameter Tuning**: Add GridSearchCV or Bayesian optimization
- **Visualization**: Create interactive dashboards
- **Documentation**: Improve explanations or add tutorials
- **Testing**: Add unit tests and integration tests
- **Deployment**: Create Docker container or REST API

---

## Roadmap

### Current Version (v1.0)
✅ Complete 7-step ML pipeline  
✅ 5 classification models  
✅ Comprehensive evaluation metrics  
✅ Automated reporting  

### Planned Features (v1.1)
- [ ] Threshold optimization for custom Recall/Precision balance
- [ ] Implementation of `class_weight='balanced'` for all models
- [ ] SMOTE for handling class imbalance
- [ ] Cross-validation with StratifiedKFold

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Summary:**
- ✅ Free to use for personal and commercial projects
- ✅ Free to modify and distribute
- ✅ Must include original copyright notice
- ❌ No warranty or liability

---

## Authors

**Data Science Team**
- Project Lead: Bruno Silva
- Contributors: [List contributors]

**Contact:**
- Email: [bruno_m_c_silva@proton.me]
- GitHub: [@brunomigueldasilva](https://github.com/brunomigueldasilva)

---

## Acknowledgments

### Inspiration
- **Home Assistant Community** - For the amazing open-source home automation platform
- **Scikit-learn Developers** - For the excellent machine learning library
- **Data Science Community** - For countless tutorials and resources

### References
1. **Home Assistant Documentation** - https://www.home-assistant.io/docs/

### Tools & Libraries
- **Python** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **Matplotlib & Seaborn** - Data visualization
- **Home Assistant** - Smart home platform

### Special Thanks
- Thank you to the open-source community for making projects like this possible
- Thanks to everyone who contributes to improving this project

---

## Project Status

**Current Status**: 🟢 Active Development

This project is actively maintained and accepting contributions. New features are regularly added based on community feedback and emerging best practices in ML.

**Last Updated**: October 2025

---

## Support

If you encounter any issues or have questions:

1. **Check the Documentation**: Read this README thoroughly
2. **Search Issues**: Look for similar issues in the [issue tracker](https://github.com/brunomigueldasilva/home-assistant-in-bed-sensor/issues)
3. **Ask Questions**: Open a new issue with the `question` label
4. **Report Bugs**: Open an issue with detailed information:
   - Python version
   - Operating system
   - Error messages
   - Steps to reproduce

**Community Support:**
- GitHub Discussions: [Project Discussions](https://github.com/brunomigueldasilva/home-assistant-in-bed-sensor/discussions)
- Email: [bruno_m_c_silva@proton.me]

---

## Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

---

**Made with ❤️ for the Home Assistant and Data Science communities**
