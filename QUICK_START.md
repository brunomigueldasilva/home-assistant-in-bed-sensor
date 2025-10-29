# Quick Start Guide

## Get Running in 5 Minutes

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn colorama
```

### 2. Prepare Your Data
Place your CSV files in an `inputs/` folder:
```
inputs/
├── bedroom_blinds.csv
├── hallway_light.csv
├── bedoroom_light.csv
├── wc_light.csv
├── bedroom_tv.csv
├── hallway_motion_sensor.csv
└── sleep.csv
```

### 3. Run the Pipeline
```bash
python 08_orchestrator.py --all
```

That's it! ✅

---

## What Happens Next?

The system will automatically:
1. ✅ Load and analyze your sensor data
2. ✅ Preprocess and prepare features
3. ✅ Train 5 different ML models
4. ✅ Evaluate performance metrics
5. ✅ Generate confusion matrices
6. ✅ Create ROC curves
7. ✅ Build a comprehensive report

**Total time**: ~5-10 minutes depending on dataset size

---

## Expected Outputs

After completion, you'll find:

```
📊 dataset.csv                    # Consolidated dataset
📁 data_processed/                # Preprocessed data & models
📁 models/                        # Trained ML models
📁 predictions/                   # Model predictions
📁 outputs/                       # Visualizations (confusion matrix, ROC)
📄 comparative_metrics.csv        # Performance comparison
📄 FINAL_REPORT.md                # Complete analysis report
📄 execution.log                  # Execution log
```

---

## View Results

Open `FINAL_REPORT.md` to see:
- Model performance comparison
- Best model recommendation
- Confusion matrices
- ROC curves
- Next steps and recommendations

---

## Common Issues

### Issue: Missing CSV files
**Solution**: Make sure all 7 CSV files are in the `inputs/` folder

### Issue: Import errors
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Memory errors
**Solution**: Reduce dataset size or increase available RAM

---

## Next Steps

1. Review `FINAL_REPORT.md` for insights
2. Check which model performed best
3. Consider implementing recommendations
4. Integrate with Home Assistant

---

## Need Help?

- 📖 Read the full [README.md](README.md)
- 🐛 Report issues on GitHub
- 💬 Join discussions

---

**Happy analyzing! 🚀**
