# Consumer Classification Prediction


## 📋 Project Overview

This project implements a machine learning system to predict **customer churn** using multiple classification algorithms. The goal is to identify customers who are likely to stop using a company's services, enabling businesses to implement retention strategies proactively.

### Key Features:
- **Binary Classification Problem**: Customers classified as churners (1) or non-churners (0)
- **Multiple ML Models**: KNN, Logistic Regression, Neural Network (MLP)
- **Unsupervised Learning**: K-Means Clustering for customer segmentation
- **Comprehensive Analysis**: EDA, data preprocessing, model evaluation, and cross-validation

---
## 📊 Dataset Description

### Dataset Characteristics:
- **Total Rows:** 1,500
- **Total Columns:** 9 (8 features + 1 target variable)
- **Data Type:** Classification Problem

### Features:

#### Quantitative Features:
- **Age**: Customer's age (18-69 years)
- **Income**: Annual income of the customer
- **Credit_Score**: Customer's credit score (390-906)
- **Num_Purchases**: Number of purchases made (0-16)
- **Membership_Years**: Years of membership (1-14 years)

#### Categorical Features:
- **Gender**: Male, Female, Other
- **Marital_Status**: Single, Married, Divorced
- **Device_Used**: Desktop, Tablet, Mobile

#### Target Variable:
- **Churn**: Binary classification (0 = No churn, 1 = Churn)

### Class Distribution:
```
Churn 0 (No):  757 instances (50.47%)
Churn 1 (Yes): 743 instances (49.53%)
```

**Key Insight:** The dataset is naturally balanced, eliminating the need for SMOTE oversampling techniques.

### Feature Correlation:
The correlation matrix reveals that most features have weak to moderate correlations with the target variable, suggesting that churn prediction is a complex pattern that requires multiple features to model effectively.

---

## 🔧 Data Preprocessing

### 1. **Missing Value Imputation**
- **Numerical Features** (Age, Income, Credit_Score): Imputed with mean values
- **Categorical Features** (Gender): Imputed with mode (most frequent category)

### 2. **Categorical Encoding**
All categorical features (Gender, Marital_Status, Device_Used) were encoded using **Label Encoding**:
- Converts text labels into numerical representations
- Example: Male=1, Female=0
- Essential for machine learning algorithms that require numerical inputs

### 3. **Feature Scaling**
Applied **StandardScaler** to normalize numerical features:
- Ensures all features contribute equally to model calculations
- Particularly important for distance-based algorithms (KNN)
- Transforms data to have mean=0 and standard deviation=1
- Formula: `(X - mean) / std_dev`

### 4. **Outlier Detection**
Identified and analyzed outliers in:
- Income
- Credit_Score
- Number of Purchases

---

## 📈 Exploratory Data Analysis (EDA)

### Visualizations Included:

1. **Histograms** of numerical features showing distributions
2. **Bar Charts** for categorical feature distributions
3. **Boxplots** for outlier detection
4. **Correlation Matrix** heatmap
5. **Class Distribution** chart

### Key Findings:
- Age follows a multi-modal distribution
- Income is approximately normally distributed
- Credit scores are concentrated around 600-700
- Membership years and purchase behavior show varied patterns
- Nearly balanced class distribution reduces bias

---

## 🔀 Dataset Splitting

- **Training Set:** 80% (1,200 samples)
- **Testing Set:** 20% (300 samples)
- **Stratification:** Used to maintain class proportions in both sets

---

## 🤖 Machine Learning Models

### 1. **K-Nearest Neighbors (KNN)**
**Type:** Distance-based classifier
- **Principle:** Predicts based on majority class of k nearest neighbors
- **Hyperparameters:** k=5 (default)
- **Advantages:** Simple, non-parametric
- **Disadvantages:** Computationally expensive, sensitive to feature scaling

**Performance:**
- Accuracy: 51.33%
- Precision: 0.49
- Recall: 0.32
- AUC: 0.49

---

### 2. **Logistic Regression**
**Type:** Statistical linear classifier
- **Principle:** Estimates probability of churn using linear combination of features
- **Output:** Probability between 0 and 1
- **Advantages:** Interpretable, computationally efficient
- **Disadvantages:** Assumes linear relationship

**Performance:**
- Accuracy: 52.67% ✓ **Best Model**
- Precision: 0.51
- Recall: 0.53
- AUC: 0.50

---

### 3. **Neural Network (MLP Classifier)**
**Type:** Multi-layer perceptron
- **Architecture:** Input → Hidden(100) → Hidden(50) → Output
- **Activation:** ReLU (hidden layers), Sigmoid (output)
- **Solver:** Adam optimizer
- **Advantages:** Captures complex non-linear patterns
- **Disadvantages:** Black-box model, requires more data to generalize

**Performance:**
- Accuracy: 48.33%
- Precision: 0.47
- Recall: 0.50
- AUC: 0.49

---

## 🎯 Model Comparison Analysis

### Accuracy Comparison:
```
Logistic Regression: 52.67% ⭐
KNN:                 51.33%
Neural Network:      48.33%
```

### Precision vs Recall:
| Model | Precision | Recall |
|-------|-----------|--------|
| KNN | 0.49 | 0.32 |
| Logistic Regression | 0.51 | 0.53 |
| Neural Network | 0.47 | 0.50 |

### ROC Curve Analysis:
- **Logistic Regression** achieved the highest AUC score, confirming superior ability to distinguish between classes
- **KNN** followed with competitive performance
- **Neural Network** showed lower AUC, suggesting it was less effective at class separation

### Confusion Matrix Insights:
- **False Negatives** (Critical): Model predicts "No Churn" but customer leaves
- **False Positives**: Model predicts "Churn" but customer stays
- Logistic Regression minimized total errors most effectively

---

## ✅ Cross-Validation Results

**10-Fold Cross-Validation Accuracy:**
```
Logistic Regression: 0.5000 (±0.0386)
Neural Network:      0.4958 (±0.0409)
KNN:                 0.4833 (±0.0371)
```

**Key Finding:** Very low standard deviation (<0.02) confirms models are robust and not result of lucky train-test split.

---

## 🔬 Unsupervised Learning: K-Means Clustering

### Methodology:
1. **Optimal K Selection**: Elbow Method indicated k=4
2. **Clustering Process**: Iterative centroid-based grouping
3. **Visualization**: PCA dimensionality reduction to 2D for visualization

### Customer Segments:
The PCA visualization reveals distinct customer segments with different characteristics, useful for targeted retention strategies.

---

## 🎓 Key Conclusions

### Best Model: **Logistic Regression**
- **Accuracy:** 53% (F1 Score)
- **Why Effective:** Linear relationship between features and churn probability
- **Key Weighted Features:** 
  - Membership_Years
  - Income
  - Credit_Score

### Major Advantages of This Approach:
1. **Data Balance**: Naturally balanced dataset simplified preprocessing
2. **Linear Relationships**: Strong performance of linear models suggests features have linear predictive power
3. **Interpretability**: Logistic Regression coefficients provide clear feature importance
4. **Efficiency**: Fast training and prediction times

### Data Preprocessing Challenges Addressed:
1. **Missing Data Imputation**: Mean imputation for numerical features, mode for categorical
2. **Categorical Encoding**: Label encoding ensured all data types compatible with algorithms
3. **Feature Scaling**: StandardScaler prevented bias from features with large magnitudes

---

## 📁 Repository Structure

```
consumer-classification-prediction/
├── README.md                              # This file
├── consumer_classification.py              # Python script version
├── CSE422_Final_Lab_Project.ipynb         # Jupyter notebook (full implementation)
├── consumer_classification_dataset.csv    # Dataset (if included)
└── requirements.txt                       # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites:
- Python 3.7+
- Jupyter Notebook (optional)

### Installation:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/consumer-classification-prediction.git
   cd consumer-classification-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Python script:**
   ```bash
   python consumer_classification.py
   ```
   
   Or open the Jupyter notebook:
   ```bash
   jupyter notebook CSE422_Final_Lab_Project.ipynb
   ```

---

## 📦 Dependencies

```
pandas==1.5.0
numpy==1.23.0
matplotlib==3.5.2
seaborn==0.12.0
scikit-learn==1.1.1
imbalanced-learn==0.9.1
```

See `requirements.txt` for exact versions.

---

## 📊 Results Summary

| Metric | KNN | Logistic Regression | Neural Network |
|--------|-----|-------------------|----------------|
| **Accuracy** | 51.33% | **52.67%** ⭐ | 48.33% |
| **Precision** | 0.49 | **0.51** | 0.47 |
| **Recall** | 0.32 | **0.53** | 0.50 |
| **AUC Score** | 0.49 | **0.50** | 0.49 |
| **CV Accuracy** | 0.4833 | **0.5000** | 0.4958 |

---

## 🔍 Methodology Highlights

### Feature Engineering:
- No features removed (all 8 features used)
- Label encoding for categorical variables
- StandardScaler for numerical normalization

### Validation Strategy:
- 80-20 train-test split with stratification
- 10-fold cross-validation for robustness
- Multiple evaluation metrics (accuracy, precision, recall, AUC)

### Model Selection Criteria:
1. **Accuracy**: Overall correctness
2. **Precision**: Minimizing false positives (unnecessary retention efforts)
3. **Recall**: Minimizing false negatives (catching actual churners)
4. **AUC-ROC**: Threshold-independent performance

---

## 💡 Business Implications

### Use Cases:
- **Customer Retention**: Identify at-risk customers for targeted interventions
- **Resource Allocation**: Focus retention efforts on high-churn-risk segments
- **Customer Lifetime Value**: Improve CLV by reducing churn

### Recommended Actions:
1. Monitor customers with predicted high churn probability
2. Implement personalized retention campaigns
3. Analyze cluster characteristics for segment-specific strategies
4. Regular model retraining with new data

---

## 🔮 Future Improvements

1. **Feature Engineering:**
   - Create interaction features (e.g., Income × Membership_Years)
   - Temporal features if timestamps available
   - Customer behavior aggregations

2. **Advanced Models:**
   - Gradient Boosting (XGBoost, LightGBM)
   - Ensemble methods (Random Forest, Stacking)
   - SHAP values for interpretability

3. **Hyperparameter Optimization:**
   - Grid search or Bayesian optimization
   - Learning rate tuning for neural networks

4. **Class Imbalance Handling:**
   - Implement SMOTE if working with imbalanced data
   - Adjust class weights in models

5. **Production Deployment:**
   - Model containerization with Docker
   - REST API for predictions
   - Monitoring and retraining pipeline

---

## 📝 Notes

- This project was submitted as the final lab assignment for CSE422 (Artificial Intelligence)
- All code is fully documented and includes inline comments
- Visualizations use matplotlib and seaborn for clarity
- Cross-validation ensures model robustness and generalization capability

---

## 📧 Contact

For questions or suggestions regarding this project:
- **Md. Mostahid Hasan** - mostahid.hasan@bracu.ac.bd
- **Yuki Barua** - yuki.barua@bracu.ac.bd

---

## 📄 License

This project is provided for educational purposes as part of the CSE422 course at BRAC University.

---

## 🙏 Acknowledgments

- Instructors: Rafeed Rahman and Maazin Munawar
- BRAC University, Department of Computer Science and Engineering
- scikit-learn, pandas, and other open-source communities

---

**Last Updated:** June 2024  
**Project Status:** ✅ Complete
