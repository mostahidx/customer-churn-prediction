# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Clone & Install
```bash
git clone https://github.com/yourusername/consumer-classification-prediction.git
cd consumer-classification-prediction
pip install -r requirements.txt
```

### Step 2: Run the Project
```bash
# Option A: Python Script
python consumer_classification.py

# Option B: Jupyter Notebook
jupyter notebook CSE422_Final_Lab_Project.ipynb
```

### Step 3: View Results
Check console output and visualizations for model performance metrics.

---

## 📊 What This Project Does

Predicts customer churn using machine learning to help businesses identify at-risk customers and implement retention strategies.

### Key Stats:
- **Dataset:** 1,500 customers with 8 features
- **Best Model:** Logistic Regression (52.67% accuracy)
- **Models Tested:** KNN, Logistic Regression, Neural Network
- **Clustering:** K-Means with 4 clusters

---

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `consumer_classification.py` | Standalone Python script - run directly |
| `CSE422_Final_Lab_Project.ipynb` | Interactive Jupyter notebook |
| `README.md` | Full documentation |
| `SETUP.md` | Detailed installation guide |
| `requirements.txt` | Python dependencies |

---

## 🔑 Key Findings

### Model Performance:
```
Logistic Regression: 52.67% ⭐ BEST
KNN:                 51.33%
Neural Network:      48.33%
```

### Features Used:
- Age, Income, Credit_Score
- Num_Purchases, Membership_Years
- Gender, Marital_Status, Device_Used

### Class Distribution:
- Non-Churners: 50.47%
- Churners: 49.53%
- (Naturally balanced - no oversampling needed)

---

## 🎯 Project Pipeline

```
1. Load Data
   ↓
2. Exploratory Data Analysis (EDA)
   ↓
3. Data Preprocessing
   - Handle missing values
   - Encode categorical features
   - Scale numerical features
   ↓
4. Train-Test Split (80-20)
   ↓
5. Model Training
   - KNN
   - Logistic Regression
   - Neural Network
   ↓
6. Model Evaluation
   - Accuracy, Precision, Recall
   - ROC Curves, Confusion Matrices
   - Cross-Validation
   ↓
7. Unsupervised Learning
   - K-Means Clustering
   - Customer Segmentation
```

---

## 💡 Quick Reference

### Install All Dependencies:
```bash
pip install -r requirements.txt
```

### Update Specific Package:
```bash
pip install --upgrade scikit-learn
```

### Run Tests:
```bash
python -m pytest
```

### Create Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

## 🎓 Understanding the Models

### 1. KNN (K-Nearest Neighbors)
- Simple: Looks at 5 nearest similar customers
- Accuracy: 51.33%
- Best for: Quick predictions

### 2. Logistic Regression ⭐
- Statistical: Calculates churn probability (0-1)
- Accuracy: 52.67% (BEST)
- Best for: Interpretable predictions

### 3. Neural Network
- Complex: Uses layers to find patterns
- Accuracy: 48.33%
- Best for: Complex non-linear patterns

---

## 📈 Expected Output

Running the script generates:

### Console Output:
```
===============================
CONSUMER CLASSIFICATION PREDICTION
===============================

Loading dataset...
Dataset Shape: (1500, 9)
Total Rows: 1500, Total Columns: 9

First few rows:
  Age    Income Gender  ...  Churn
0  32  45000.5   Male  ...      0
1  28  52000.2 Female  ...      1

[Processing continues...]

Model Evaluation:
- KNN Accuracy: 0.5133
- Logistic Regression Accuracy: 0.5267 ✓ BEST
- Neural Network Accuracy: 0.4833
```

### Visualizations:
- Class distribution charts
- Feature distributions
- Correlation heatmaps
- Model accuracy comparison
- ROC curves
- Confusion matrices

---

## 🐛 Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Dataset not found | Place CSV in project folder or update path |
| Jupyter won't start | `pip install --upgrade jupyter` |
| Slow performance | Use smaller dataset for testing |

---

## 📚 Learn More

- **Full Documentation:** See `README.md`
- **Setup Guide:** See `SETUP.md`
- **Data Details:** Check PDF report in uploads
- **Code Comments:** Read inline documentation in scripts

---

## 🎯 Next Steps

1. ✅ Install requirements
2. ✅ Run the script
3. ✅ Review visualizations
4. ✅ Read the README.md
5. ✅ Experiment with parameters
6. ✅ Deploy to production (advanced)

---

## 👥 Team

- **Md. Mostahid Hasan** (23301693)
- **Yuki Barua** (22241007)

**Course:** Artificial Intelligence (CSE422) - BRAC University

---

## 📞 Need Help?

1. Check `SETUP.md` for detailed instructions
2. Review `README.md` for full documentation
3. Check inline code comments
4. Review error messages carefully

---

**Happy Modeling! 🚀**

*Last Updated: June 2024*
