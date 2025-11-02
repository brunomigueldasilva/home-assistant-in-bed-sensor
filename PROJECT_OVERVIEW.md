# Project Overview: Home Assistant In Bed Sensor

## Executive Summary

This is a **production-ready machine learning pipeline** that predicts whether a person is lying in bed using smart home sensor data from Home Assistant. The project demonstrates best practices in supervised learning, data preprocessing, model evaluation, and automated reporting.

---

## Key Highlights

### 🎯 Business Value
- **Automated sleep detection** for smart home optimization
- **Energy savings** through intelligent sleep mode activation
- **Comfort enhancement** via automatic environment adjustments
- **Sleep pattern monitoring** without wearable devices

### 🔬 Technical Excellence
- **7-step automated pipeline** from data to insights
- **5 ML models** with comprehensive comparison
- **Class imbalance handling** via stratified sampling
- **Production-ready code** with orchestration and logging
- **Reproducible results** with seeded random states

### 📊 Expected Performance
- **Accuracy**: 85-95%
- **F1-Score**: 80-91%
- **ROC-AUC**: 0.90-0.98
- **Training time**: ~5-10 minutes

---

## Project Structure

```
📦 Home Assistant In Bed Sensor
│
├── 📄 README.md                 ← You are here! (Main documentation)
├── 📄 QUICK_START.md            ← 5-minute getting started guide
├── 📄 CONTRIBUTING.md           ← Developer contribution guidelines
├── 📄 LICENSE                   ← MIT License
├── 📄 requirements.txt          ← Python dependencies
│
├── 🐍 Python Scripts (7-step pipeline)
│   ├── 01_exploratory_analysis.py
│   ├── 02_preprocessing.py
│   ├── 03_train_models.py
│   ├── 04_evaluate_metrics.py
│   ├── 05_confusion_matrix.py
│   ├── 06_roc_curves.py
│   └── 07_final_report.py
│
├── 🎮 08_orchestrator.py        ← Pipeline automation & control
│
└── 📁 Data & Outputs
    ├── inputs/                   ← Your CSV sensor files
    ├── outputs/                  ← output files
    │   ├── graphics/             ← Visualizations (PNG)
    │   ├── data_processed/       ← Preprocessed datasets
    │   ├── models/               ← Trained ML models
    │   └── predictions/          ← Model predictions
    └── FINAL_REPORT.md           ← Auto-generated analysis
```

---

## The 7-Step Machine Learning Pipeline

### 🔍 Step 1: Exploratory Data Analysis
- **Purpose**: Understand data distribution and patterns
- **Script**: `01_exploratory_analysis.py`
- **Outputs**: Consolidated dataset, EDA visualizations
- **Key Activities**:
  - Load 7 CSV sensor files
  - Merge by timestamp
  - Remove invalid states
  - Create target variable
  - Generate visualizations

### 🧹 Step 2: Data Preprocessing
- **Purpose**: Prepare data for model training
- **Script**: `02_preprocessing.py`
- **Outputs**: Train/test splits, fitted scaler
- **Key Activities**:
  - One-hot encode categorical variables
  - Stratified train-test split (80/20)
  - Fit StandardScaler on training data only
  - Save preprocessed data and metadata

### 🤖 Step 3: Model Training
- **Purpose**: Train multiple classification models
- **Script**: `03_train_models.py`
- **Outputs**: 5 trained models, predictions
- **Models Trained**:
  1. Logistic Regression
  2. K-Nearest Neighbors
  3. SVM Linear
  4. SVM RBF
  5. Naïve Bayes

### 📈 Step 4: Metrics Evaluation
- **Purpose**: Calculate performance metrics
- **Script**: `04_evaluate_metrics.py`
- **Outputs**: Comparative metrics table
- **Metrics Calculated**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

### 🎯 Step 5: Confusion Matrix
- **Purpose**: Visualize classification errors
- **Script**: `05_confusion_matrix.py`
- **Outputs**: Confusion matrix visualization
- **Analysis**: True/False Positives/Negatives

### 📉 Step 6: ROC Curves
- **Purpose**: Compare discrimination ability
- **Script**: `06_roc_curves.py`
- **Outputs**: ROC curves, AUC comparison
- **Insight**: Model threshold optimization

### 📝 Step 7: Final Report
- **Purpose**: Generate comprehensive analysis
- **Script**: `07_final_report.py`
- **Outputs**: Markdown report with all insights
- **Includes**: Recommendations and next steps

---

## Quick Usage Examples

### Run Everything (Automated)
```bash
python 08_orchestrator.py --all
```

### Interactive Mode
```bash
python 08_orchestrator.py
# Choose from menu: Run all, specific steps, or clean outputs
```

### Run Specific Steps
```bash
python 08_orchestrator.py --steps 1,3,7
```

### Individual Scripts
```bash
python 01_exploratory_analysis.py
python 02_preprocessing.py
# ... and so on
```

---

## Data Requirements

### Input Files (in `inputs/` folder)
1. **bedroom_blinds.csv** - Bedroom blinds sensor
2. **hallway_light.csv** - Hallway light sensor
3. **bedoroom_light.csv** - Bedroom light sensor
4. **wc_light.csv** - Bathroom light sensor
5. **bedroom_tv.csv** - Bedroom TV sensor
6. **hallway_motion_sensor.csv** - Hallway motion sensor
7. **sleep.csv** - Sleep tracking ground truth

### CSV Format
```csv
last_changed,state,sensor
2024-01-15 22:30:15,off,bedroom_light
2024-01-15 23:15:42,on,motion_sensor
```

---

## Expected Outputs

After running the complete pipeline:

```
✅ outputs/dataset.csv             # Consolidated sensor data
✅ outputs/data_processed/         # Preprocessed train/test sets
   ├── X_train.pkl
   ├── X_test.pkl
   ├── y_train.pkl
   ├── y_test.pkl
   └── scaler.pkl

✅ outputs/models/                 # 5 trained models
   ├── logistic_regression.pkl
   ├── knn.pkl
   ├── svm_linear.pkl
   ├── svm_rbf.pkl
   └── naive_bayes.pkl

✅ outputs/predictions/            # Model predictions
   ├── logistic_regression_predictions.csv
   └── ... (all models)

✅ outputs/                        # Visualizations
   ├── confusion_matrix.png
   └── roc_curves.png

✅ comparative_metrics.csv         # Performance comparison
✅ auc_comparison.csv              # ROC-AUC scores
✅ training_times.csv              # Training durations
✅ FINAL_REPORT.md                 # Complete analysis
✅ execution.log                   # Execution log
```

---

## Documentation Files

| File | Purpose | Lines | Size |
|------|---------|-------|------|
| **README.md** | Main project documentation | 642 | 19 KB |
| **QUICK_START.md** | 5-minute getting started | 90 | 2.3 KB |
| **CONTRIBUTING.md** | Contribution guidelines | 386 | 9 KB |
| **requirements.txt** | Python dependencies | 35 | 644 B |
| **LICENSE** | MIT License | 21 | 1.1 KB |

---

## Technology Stack

### Core Technologies
- **Language**: Python 3.8+
- **ML Library**: scikit-learn 1.0+
- **Data Processing**: pandas, NumPy
- **Visualization**: matplotlib, seaborn
- **Platform**: Home Assistant

### Development Tools
- **Orchestration**: Custom Python orchestrator
- **Logging**: Built-in Python logging
- **Serialization**: pickle
- **Testing**: pytest (optional)

---

## Performance Characteristics

### Computational Requirements
- **Memory**: ~500 MB - 2 GB (depends on dataset size)
- **CPU**: Any modern processor
- **Time**: 5-10 minutes for complete pipeline
- **Storage**: ~10 MB for models and outputs

### Scalability
- ✅ Handles thousands of records efficiently
- ✅ Modular design allows easy extension
- ⚠️ Very large datasets (>1M records) may require optimization

---

## Model Comparison Summary

| Model | Speed | Interpretability | Non-linear | Best For |
|-------|-------|------------------|------------|----------|
| **Logistic Regression** | ⚡⚡⚡ | ⭐⭐⭐ | ❌ | Baseline, interpretability |
| **KNN** | ⚡⚡ | ⭐⭐ | ✅ | Local patterns |
| **SVM Linear** | ⚡⚡ | ⭐⭐ | ❌ | High-dimensional data |
| **SVM RBF** | ⚡ | ⭐ | ✅ | Complex boundaries |
| **Naïve Bayes** | ⚡⚡⚡ | ⭐⭐ | ❌ | Quick baseline |

---

## Common Use Cases

### 1. Home Automation Developer
**Goal**: Integrate sleep detection into smart home system
**Steps**:
1. Collect sensor data from Home Assistant
2. Run pipeline to train model
3. Deploy best model to production
4. Trigger automations based on predictions

### 2. Data Science Student
**Goal**: Learn ML pipeline development
**Steps**:
1. Study the 7-step workflow
2. Understand preprocessing techniques
3. Compare model performance
4. Experiment with hyperparameters

### 3. Sleep Researcher
**Goal**: Analyze sleep patterns
**Steps**:
1. Use provided visualizations
2. Study temporal patterns in data
3. Identify bedtime routine indicators
4. Generate insights from final report

### 4. Energy Efficiency Consultant
**Goal**: Optimize home energy usage
**Steps**:
1. Analyze "in bed" detection accuracy
2. Calculate potential energy savings
3. Recommend automation strategies
4. Implement sleep mode activation

---

## Advantages of This Project

### ✅ Complete Solution
- End-to-end pipeline from raw data to insights
- No missing steps or incomplete workflows
- Ready to use out of the box

### ✅ Best Practices
- Stratified splitting for imbalanced data
- Prevents data leakage (scaler fitted on train only)
- Reproducible results (seeded random states)
- Comprehensive evaluation metrics

### ✅ Production Ready
- Automated orchestration
- Error handling and logging
- Modular and maintainable code
- Clear documentation

### ✅ Educational Value
- Clear structure demonstrates ML workflow
- Comments explain why decisions were made
- Suitable for learning and teaching
- Multiple models for comparison

---

## Limitations & Future Work

### Current Limitations
- ⚠️ Limited to binary classification
- ⚠️ No hyperparameter tuning
- ⚠️ Single-user system
- ⚠️ No real-time prediction API

### Planned Improvements (See Roadmap)
- [ ] Threshold optimization
- [ ] SMOTE for class imbalance
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Ensemble methods (Random Forest, XGBoost)
- [ ] Real-time API
- [ ] Multi-user support
- [ ] Web dashboard

---

## Success Metrics

After running this project, you will have:

1. ✅ **Trained Models**: 5 different classifiers
2. ✅ **Performance Metrics**: Comprehensive evaluation
3. ✅ **Visualizations**: Confusion matrix and ROC curves
4. ✅ **Best Model**: Identified based on F1-Score
5. ✅ **Insights**: Business recommendations
6. ✅ **Reproducibility**: Saved models for deployment
7. ✅ **Documentation**: Complete analysis report

---

## Getting Started

### For Beginners
1. Read [QUICK_START.md](QUICK_START.md) first
2. Follow step-by-step installation
3. Run `python 08_orchestrator.py --all`
4. Review `FINAL_REPORT.md`

### For Developers
1. Read [README.md](README.md) thoroughly
2. Review [CONTRIBUTING.md](CONTRIBUTING.md)
3. Study individual scripts (01-07)
4. Experiment with parameters

### For Researchers
1. Examine preprocessing pipeline
2. Study model comparison methodology
3. Analyze evaluation metrics
4. Build upon this foundation

---

## Support & Community

### Get Help
- 📖 Read the documentation
- 🔍 Search existing issues
- 💬 Start a discussion
- 📧 Contact maintainers

### Stay Updated
- ⭐ Star the repository
- 👀 Watch for updates
- 🍴 Fork and customize
- 🤝 Contribute improvements

---

## License

**MIT License** - Free to use, modify, and distribute. See [LICENSE](LICENSE) file.

---

## Credits

**Created by**: Data Science Team  
**Purpose**: Demonstrate supervised learning best practices  
**Platform**: Home Assistant + Python + scikit-learn  
**Status**: 🟢 Active Development

---

## Final Notes

This project serves as both:
1. **A practical tool** for home automation
2. **An educational resource** for learning ML

Whether you're implementing this in your smart home or using it to learn machine learning, we hope you find it valuable!

**Questions?** Check the [README.md](README.md) or open an issue on GitHub.

---

**Happy Learning & Automating! 🏠🤖📊**
