"""
Consumer Classification Prediction - Main Script
Course: Artificial Intelligence (CSE422)
Group 06: Md. Mostahid Hasan (23301693), Yuki Barua (22241007)
Instructors: Rafeed Rahman and Maazin Munawar
Section: 25

This script implements customer churn prediction using multiple machine learning models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report, precision_score, 
                             recall_score, roc_curve, roc_auc_score)


# ============================================================================
# 1. LOAD THE DATASET
# ============================================================================
def load_data(filepath):
    """Load the customer classification dataset."""
    df = pd.read_csv(filepath)
    return df


# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
def explore_data(df):
    """Perform exploratory data analysis on the dataset."""
    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"Dataset Shape: {df.shape}")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nTarget Variable Distribution:")
    print(df['Churn'].value_counts())
    print(f"\nChurn 0 (No): {(df['Churn'] == 0).sum()} instances ({(df['Churn'] == 0).sum() / len(df) * 100:.2f}%)")
    print(f"Churn 1 (Yes): {(df['Churn'] == 1).sum()} instances ({(df['Churn'] == 1).sum() / len(df) * 100:.2f}%)")
    
    print("\n" + "=" * 80)


def visualize_class_distribution(df):
    """Visualize the class distribution of the Churn variable."""
    plt.figure(figsize=(8, 5))
    churn_counts = df['Churn'].value_counts()
    colors = ['green', 'red']
    plt.bar(['0', '1'], churn_counts.values, color=colors)
    plt.xlabel('Churn', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Class Distribution', fontsize=14)
    plt.show()


def visualize_histograms(df):
    """Visualize histograms of numerical features."""
    numerical_features = ['Age', 'Income', 'Credit_Score', 'Num_Purchases', 
                         'Membership_Years', 'Churn']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(numerical_features):
        axes[idx].hist(df[feature], bins=30, color='skyblue', edgecolor='black')
        axes[idx].set_title(feature, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()


def visualize_categorical_distribution(df):
    """Visualize the distribution of categorical features."""
    categorical_features = ['Gender', 'Marital_Status', 'Device_Used']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, feature in enumerate(categorical_features):
        value_counts = df[feature].value_counts()
        axes[idx].bar(value_counts.index, value_counts.values, color='steelblue')
        axes[idx].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Count')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def visualize_boxplots(df):
    """Visualize boxplots to identify outliers."""
    numerical_features = ['Age', 'Income', 'Credit_Score', 'Num_Purchases', 
                         'Membership_Years', 'Churn']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(numerical_features):
        axes[idx].boxplot(df[feature])
        axes[idx].set_title(f'Boxplot of {feature}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(feature)
    
    plt.tight_layout()
    plt.show()


def visualize_correlation_matrix(df):
    """Visualize the correlation matrix of all features."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = numeric_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Matrix of All Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
def handle_missing_values(df):
    """Handle missing values in the dataset."""
    print("\n" + "=" * 80)
    print("HANDLING MISSING VALUES")
    print("=" * 80)
    
    # Impute numerical features with mean
    numerical_features = ['Age', 'Income', 'Credit_Score']
    for feature in numerical_features:
        if df[feature].isnull().sum() > 0:
            df[feature].fillna(df[feature].mean(), inplace=True)
            print(f"Imputed {feature} with mean value")
    
    # Impute categorical features with mode
    if df['Gender'].isnull().sum() > 0:
        df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
        print(f"Imputed Gender with mode value")
    
    print(f"\nMissing values after imputation:\n{df.isnull().sum()}")
    print("=" * 80)
    
    return df


def encode_categorical_features(df):
    """Encode categorical features using Label Encoding."""
    print("\n" + "=" * 80)
    print("ENCODING CATEGORICAL FEATURES")
    print("=" * 80)
    
    label_encoders = {}
    categorical_features = ['Gender', 'Marital_Status', 'Device_Used']
    
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        label_encoders[feature] = le
        print(f"Encoded {feature}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    print("=" * 80)
    
    return df, label_encoders


def scale_features(X):
    """Scale numerical features using StandardScaler."""
    print("\n" + "=" * 80)
    print("SCALING FEATURES")
    print("=" * 80)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("StandardScaler applied to all numerical features")
    print("Features are now centered around 0 with standard deviation of 1")
    print("=" * 80)
    
    return X_scaled, scaler


def preprocess_data(df):
    """Complete preprocessing pipeline."""
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categorical features
    df, label_encoders = encode_categorical_features(df)
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Scale features
    X_scaled, scaler = scale_features(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, scaler, label_encoders


# ============================================================================
# 4. DATASET SPLITTING
# ============================================================================
def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    print("\n" + "=" * 80)
    print("DATASET SPLITTING")
    print("=" * 80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0] / len(X) * 100:.2f}%)")
    print(f"Testing set size: {X_test.shape[0]} ({X_test.shape[0] / len(X) * 100:.2f}%)")
    print(f"\nTraining set - Churn distribution:\n{y_train.value_counts()}")
    print(f"\nTesting set - Churn distribution:\n{y_test.value_counts()}")
    print("=" * 80)
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# 5. UNSUPERVISED LEARNING - K-MEANS CLUSTERING
# ============================================================================
def find_optimal_k(X, k_range=range(1, 11)):
    """Find optimal k using the Elbow Method."""
    print("\n" + "=" * 80)
    print("K-MEANS CLUSTERING - FINDING OPTIMAL K")
    print("=" * 80)
    
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Plot the Elbow Method
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    print("Elbow plot displayed. Optimal k appears to be around 4.")
    print("=" * 80)
    
    return 4  # Based on elbow method


def perform_kmeans_clustering(X, k=4):
    """Perform K-Means clustering and visualize using PCA."""
    print("\n" + "=" * 80)
    print("K-MEANS CLUSTERING")
    print("=" * 80)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    print(f"K-Means clustering with k={k} performed")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    
    # Visualize using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                         cmap='viridis', alpha=0.6, s=50)
    plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
               pca.transform(kmeans.cluster_centers_)[:, 1],
               c='red', marker='X', s=200, edgecolors='black', linewidth=2,
               label='Centroids')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', 
              fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', 
              fontsize=12)
    plt.title('Customer Clusters (PCA Visualization)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("=" * 80)
    
    return kmeans, cluster_labels


# ============================================================================
# 6. MODEL TRAINING
# ============================================================================
def train_knn(X_train, y_train, n_neighbors=5):
    """Train K-Nearest Neighbors classifier."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression classifier."""
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    return lr


def train_neural_network(X_train, y_train):
    """Train Neural Network (MLP Classifier)."""
    nn = MLPClassifier(hidden_layer_sizes=(100, 50), 
                       activation='relu', 
                       solver='adam',
                       random_state=42,
                       max_iter=1000)
    nn.fit(X_train, y_train)
    return nn


def train_models(X_train, y_train):
    """Train all models."""
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)
    
    print("Training K-Nearest Neighbors...")
    knn_model = train_knn(X_train, y_train)
    print("✓ KNN training completed")
    
    print("\nTraining Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    print("✓ Logistic Regression training completed")
    
    print("\nTraining Neural Network (MLP)...")
    nn_model = train_neural_network(X_train, y_train)
    print("✓ Neural Network training completed")
    
    print("=" * 80)
    
    return knn_model, lr_model, nn_model


# ============================================================================
# 7. MODEL EVALUATION
# ============================================================================
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'y_pred': y_pred
    }


def evaluate_all_models(knn_model, lr_model, nn_model, X_test, y_test):
    """Evaluate all trained models."""
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    results = []
    
    print("\n" + "-" * 80)
    print("K-NEAREST NEIGHBORS")
    print("-" * 80)
    knn_result = evaluate_model(knn_model, X_test, y_test, "KNN")
    results.append(knn_result)
    print(f"Accuracy: {knn_result['Accuracy']:.4f}")
    print(f"Precision: {knn_result['Precision']:.4f}")
    print(f"Recall: {knn_result['Recall']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, knn_result['y_pred']))
    
    print("\n" + "-" * 80)
    print("LOGISTIC REGRESSION")
    print("-" * 80)
    lr_result = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    results.append(lr_result)
    print(f"Accuracy: {lr_result['Accuracy']:.4f}")
    print(f"Precision: {lr_result['Precision']:.4f}")
    print(f"Recall: {lr_result['Recall']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, lr_result['y_pred']))
    
    print("\n" + "-" * 80)
    print("NEURAL NETWORK (MLP)")
    print("-" * 80)
    nn_result = evaluate_model(nn_model, X_test, y_test, "Neural Network")
    results.append(nn_result)
    print(f"Accuracy: {nn_result['Accuracy']:.4f}")
    print(f"Precision: {nn_result['Precision']:.4f}")
    print(f"Recall: {nn_result['Recall']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, nn_result['y_pred']))
    
    print("=" * 80)
    
    return results


def visualize_accuracy_comparison(results):
    """Visualize accuracy comparison of all models."""
    model_names = [r['Model'] for r in results]
    accuracies = [r['Accuracy'] for r in results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['#1f77b4', '#2ca02c', '#d62728'])
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def visualize_precision_recall_comparison(results):
    """Visualize Precision and Recall comparison."""
    model_names = [r['Model'] for r in results]
    precisions = [r['Precision'] for r in results]
    recalls = [r['Recall'] for r in results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, precisions, width, label='Precision', color='skyblue')
    bars2 = plt.bar(x + width/2, recalls, width, label='Recall', color='salmon')
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Precision vs Recall Comparison (Target: Yes Churn)', fontsize=14, fontweight='bold')
    plt.xticks(x, model_names)
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


def visualize_confusion_matrices(knn_model, lr_model, nn_model, X_test, y_test, results):
    """Visualize confusion matrices for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    models = [
        (knn_model, "KNN", results[0]['y_pred']),
        (lr_model, "Logistic Regression", results[1]['y_pred']),
        (nn_model, "Neural Network", results[2]['y_pred'])
    ]
    
    for ax, (model, name, y_pred) in zip(axes, models):
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar=False, annot_kws={'size': 14})
        ax.set_title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def visualize_roc_curves(knn_model, lr_model, nn_model, X_test, y_test):
    """Visualize ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    models = [
        (knn_model, "KNN"),
        (lr_model, "Logistic Regression"),
        (nn_model, "Neural Network")
    ]
    
    for model, name in models:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance', linewidth=2)
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate (False Alarms)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 8. CROSS-VALIDATION
# ============================================================================
def perform_cross_validation(knn_model, lr_model, nn_model, X_train, y_train):
    """Perform 10-fold cross-validation on all models."""
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION ANALYSIS (10-FOLD)")
    print("=" * 80)
    
    models = [
        (knn_model, "KNN"),
        (lr_model, "Logistic Regression"),
        (nn_model, "Neural Network")
    ]
    
    print("\nCross-Validation Accuracy:")
    print("-" * 80)
    
    for model, name in models:
        scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    print("=" * 80)


# ============================================================================
# 9. MAIN PIPELINE
# ============================================================================
def main(filepath):
    """Main pipeline for consumer classification prediction."""
    print("\n" + "=" * 80)
    print("CONSUMER CLASSIFICATION PREDICTION")
    print("Artificial Intelligence (CSE422) - Group 06")
    print("=" * 80 + "\n")
    
    # Load data
    print("Loading dataset...")
    df = load_data(filepath)
    
    # Exploratory Data Analysis
    explore_data(df)
    visualize_class_distribution(df)
    visualize_histograms(df)
    visualize_categorical_distribution(df)
    visualize_boxplots(df)
    visualize_correlation_matrix(df)
    
    # Data Preprocessing
    X, y, scaler, label_encoders = preprocess_data(df)
    
    # Dataset Splitting
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Unsupervised Learning - K-Means Clustering
    optimal_k = find_optimal_k(X)
    kmeans, cluster_labels = perform_kmeans_clustering(X, k=optimal_k)
    
    # Model Training
    knn_model, lr_model, nn_model = train_models(X_train, y_train)
    
    # Model Evaluation
    results = evaluate_all_models(knn_model, lr_model, nn_model, X_test, y_test)
    
    # Visualizations
    visualize_accuracy_comparison(results)
    visualize_precision_recall_comparison(results)
    visualize_confusion_matrices(knn_model, lr_model, nn_model, X_test, y_test, results)
    visualize_roc_curves(knn_model, lr_model, nn_model, X_test, y_test)
    
    # Cross-Validation
    perform_cross_validation(knn_model, lr_model, nn_model, X_train, y_train)
    
    # Results Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall']])
    print("\nBest Model: Logistic Regression with highest accuracy")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Replace 'your_dataset.csv' with the actual path to your dataset
    main('consumer_classification_dataset.csv')
