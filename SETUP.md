# Setup Guide - Consumer Classification Prediction

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Running the Project](#running-the-project)
4. [Troubleshooting](#troubleshooting)
5. [Project Structure](#project-structure)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.7 or higher** - [Download Python](https://www.python.org/downloads/)
- **Git** - [Download Git](https://git-scm.com/downloads)
- **pip** (Python package manager) - Usually comes with Python

### Verify Installation:
```bash
python --version
pip --version
git --version
```

---

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/consumer-classification-prediction.git
cd consumer-classification-prediction
```

### Step 2: Create a Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- pandas - Data manipulation
- numpy - Numerical computing
- matplotlib - Data visualization
- seaborn - Statistical visualization
- scikit-learn - Machine learning
- imbalanced-learn - Handling imbalanced data
- jupyter - Notebook interface

---

## Running the Project

### Option 1: Run the Python Script

```bash
python consumer_classification.py
```

**Note:** Make sure your dataset file (`consumer_classification_dataset.csv`) is in the same directory or update the file path in the script.

### Option 2: Run with Jupyter Notebook

```bash
jupyter notebook CSE422_Final_Lab_Project.ipynb
```

This will open the notebook in your default web browser where you can:
- Run cells individually
- View outputs and visualizations
- Modify and experiment with code

---

## Data Preparation

### Dataset Requirements:

Your dataset should have the following columns:
- `Age` (numeric)
- `Income` (numeric)
- `Gender` (categorical: Male, Female, Other)
- `Marital_Status` (categorical: Single, Married, Divorced)
- `Credit_Score` (numeric)
- `Num_Purchases` (numeric)
- `Membership_Years` (numeric)
- `Device_Used` (categorical: Desktop, Mobile, Tablet)
- `Churn` (binary: 0 or 1)

### File Format:
- CSV (.csv) format recommended
- Ensure no header issues or encoding problems
- File should have 1,500+ rows for best results

### Loading Data:

**If using the Python script:**
```python
# Modify the filepath in main()
main('path/to/your/consumer_classification_dataset.csv')
```

**If using the Notebook:**
```python
df = pd.read_csv('path/to/your/consumer_classification_dataset.csv')
```

---

## Output Files

After running the project, you will generate:

### Visualizations:
- **Class Distribution Chart**
- **Feature Histograms** (Age, Income, Credit_Score, etc.)
- **Categorical Distribution Plots**
- **Boxplots for Outlier Detection**
- **Correlation Matrix Heatmap**
- **Model Accuracy Comparison**
- **Precision vs Recall Comparison**
- **Confusion Matrices**
- **ROC Curves**

### Console Output:
- Data summary statistics
- Missing value counts
- Model performance metrics
- Cross-validation scores
- Classification reports

---

## Troubleshooting

### Issue 1: `ModuleNotFoundError`

**Problem:** "No module named 'pandas'" or similar

**Solution:**
```bash
pip install --upgrade pandas numpy scikit-learn
```

### Issue 2: Dataset File Not Found

**Problem:** `FileNotFoundError: consumer_classification_dataset.csv not found`

**Solution:**
1. Check the file exists in your project directory
2. Verify the exact filename
3. Update the filepath in the script:
```python
main('/absolute/path/to/dataset.csv')  # Use absolute path
```

### Issue 3: Jupyter Kernel Issues

**Problem:** Kernel dies or notebook won't run

**Solution:**
```bash
pip install --upgrade jupyter
jupyter kernel install --user
```

### Issue 4: Memory Issues with Large Datasets

**Problem:** MemoryError when processing large files

**Solution:**
```python
# Process data in chunks
chunk_size = 10000
for chunk in pd.read_csv('file.csv', chunksize=chunk_size):
    # Process each chunk
    pass
```

### Issue 5: Import Errors in Colab

If running in Google Colab, install packages directly:
```python
!pip install scikit-learn pandas numpy matplotlib seaborn imbalanced-learn
```

---

## Project Structure

```
consumer-classification-prediction/
│
├── README.md                              # Main project documentation
├── SETUP.md                               # This file
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore file
│
├── consumer_classification.py              # Standalone Python script
│                                          # (Can run independently)
│
├── CSE422_Final_Lab_Project.ipynb         # Jupyter Notebook version
│                                          # (For interactive exploration)
│
└── consumer_classification_dataset.csv    # Dataset (if included)
                                          # (1,500 rows, 9 columns)
```

---

## File Descriptions

### `consumer_classification.py`
- **Purpose:** Standalone Python implementation
- **Usage:** Run directly with `python consumer_classification.py`
- **Advantages:** Faster execution, suitable for automation
- **Structure:** Modular functions for each pipeline stage

### `CSE422_Final_Lab_Project.ipynb`
- **Purpose:** Interactive Jupyter notebook
- **Usage:** Run with `jupyter notebook`
- **Advantages:** Step-by-step execution, inline visualizations
- **Best For:** Learning, experimentation, documentation

### `README.md`
- Comprehensive project documentation
- Dataset description
- Model explanations
- Results and findings
- Business implications

### `requirements.txt`
- List of all required Python packages
- Specific versions for reproducibility
- Easy setup with single command

### `.gitignore`
- Specifies files to ignore in Git
- Prevents large files, temporary files from being committed
- Keeps repository clean

---

## Running Specific Sections

### If you only want to:

**Load and explore data:**
```python
df = load_data('consumer_classification_dataset.csv')
explore_data(df)
visualize_histograms(df)
```

**Preprocess data:**
```python
X, y, scaler, encoders = preprocess_data(df)
```

**Train models only:**
```python
X_train, X_test, y_train, y_test = split_data(X, y)
knn, lr, nn = train_models(X_train, y_train)
```

**Evaluate specific model:**
```python
results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
```

---

## Performance Optimization

### For Faster Execution:

1. **Reduce dataset size for testing:**
```python
df_sample = df.sample(frac=0.1, random_state=42)  # Use 10% of data
```

2. **Skip visualizations:**
Comment out visualization functions in main()

3. **Use subset of models:**
Train only KNN and Logistic Regression

4. **Reduce cross-validation folds:**
```python
cross_val_score(model, X_train, y_train, cv=5)  # Use 5 folds instead of 10
```

---

## System Requirements

### Minimum:
- RAM: 2 GB
- Disk Space: 500 MB (with Python and packages)
- Processor: Any modern CPU

### Recommended:
- RAM: 4+ GB
- Disk Space: 2 GB
- Processor: Multi-core CPU (for faster processing)

---

## Getting Help

### Documentation:
- [pandas Documentation](https://pandas.pydata.org/)
- [scikit-learn Guide](https://scikit-learn.org/stable/)
- [matplotlib Tutorial](https://matplotlib.org/stable/tutorials/index.html)

### Common Errors:
- Check Python version compatibility
- Ensure all dependencies installed
- Verify data file format and location
- Check console output for specific error messages

---

## Tips for Success

1. **Always use a virtual environment** to avoid package conflicts
2. **Keep your dataset in the project directory** for easier access
3. **Start with small datasets** to test the pipeline
4. **Review the README** before running for context
5. **Examine visualizations** carefully for insights
6. **Experiment with parameters** to improve results

---

## Next Steps

After successful setup:

1. **Explore the data** using visualizations
2. **Understand each model's** behavior
3. **Compare performance metrics** across models
4. **Modify hyperparameters** for improvement
5. **Deploy to production** (advanced)

---

**For questions or issues, refer to the README.md or contact the project authors.**

Happy learning! 🚀
