import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_loan_data(n_samples=10000):
    """
    Generate synthetic loan default data similar to the "Give Me Some Credit" dataset
    """
    
    # Generate base features using make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],  # Imbalanced dataset (10% default rate)
        flip_y=0.01,
        random_state=42
    )
    
    # Create meaningful feature names and transform to realistic ranges
    feature_names = [
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ]
    
    # Transform features to realistic ranges
    data = pd.DataFrame(X, columns=feature_names)
    
    # RevolvingUtilizationOfUnsecuredLines (0-1, some outliers)
    data['RevolvingUtilizationOfUnsecuredLines'] = np.abs(data['RevolvingUtilizationOfUnsecuredLines']) * 0.3
    data.loc[np.random.choice(data.index, size=int(0.05 * len(data)), replace=False), 
             'RevolvingUtilizationOfUnsecuredLines'] *= 10  # Add some outliers
    
    # Age (18-100)
    data['age'] = np.clip(np.abs(data['age']) * 15 + 35, 18, 100).astype(int)
    
    # Past due counts (0-10, mostly 0)
    for col in ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 
                'NumberOfTime60-89DaysPastDueNotWorse']:
        data[col] = np.clip(np.abs(data[col]) * 0.5, 0, 10).astype(int)
        # Make most values 0
        data.loc[data[col] < 1, col] = 0
    
    # DebtRatio (0-2, some outliers)
    data['DebtRatio'] = np.abs(data['DebtRatio']) * 0.5
    data.loc[np.random.choice(data.index, size=int(0.02 * len(data)), replace=False), 
             'DebtRatio'] *= 20  # Add some extreme outliers
    
    # MonthlyIncome (1000-15000, with some missing values)
    data['MonthlyIncome'] = np.abs(data['MonthlyIncome']) * 2000 + 3000
    # Add some missing values
    missing_idx = np.random.choice(data.index, size=int(0.2 * len(data)), replace=False)
    data.loc[missing_idx, 'MonthlyIncome'] = np.nan
    
    # NumberOfOpenCreditLinesAndLoans (0-30)
    data['NumberOfOpenCreditLinesAndLoans'] = np.clip(np.abs(data['NumberOfOpenCreditLinesAndLoans']) * 3 + 5, 0, 30).astype(int)
    
    # NumberRealEstateLoansOrLines (0-10)
    data['NumberRealEstateLoansOrLines'] = np.clip(np.abs(data['NumberRealEstateLoansOrLines']) * 1.5, 0, 10).astype(int)
    
    # NumberOfDependents (0-10, with some missing values)
    data['NumberOfDependents'] = np.clip(np.abs(data['NumberOfDependents']) * 1.2, 0, 10).astype(int)
    # Add some missing values
    missing_idx = np.random.choice(data.index, size=int(0.05 * len(data)), replace=False)
    data.loc[missing_idx, 'NumberOfDependents'] = np.nan
    
    # Add target variable
    data['SeriousDlqin2yrs'] = y
    
    return data

def save_datasets():
    """Generate and save training and test datasets"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate training data
    print("Generating training dataset...")
    train_data = generate_synthetic_loan_data(n_samples=150000)
    train_data.to_csv('data/cs-training.csv', index=False)
    print(f"Training dataset saved: {train_data.shape}")
    
    # Generate test data (without target variable)
    print("Generating test dataset...")
    test_data = generate_synthetic_loan_data(n_samples=101503)
    test_data_features = test_data.drop('SeriousDlqin2yrs', axis=1)
    test_data_features.to_csv('data/cs-test.csv', index=False)
    print(f"Test dataset saved: {test_data_features.shape}")
    
    # Save test labels separately for evaluation
    test_labels = test_data[['SeriousDlqin2yrs']]
    test_labels.to_csv('data/cs-test-labels.csv', index=False)
    
    print("\nDataset generation completed!")
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data_features.shape}")
    print(f"Default rate in training: {train_data['SeriousDlqin2yrs'].mean():.3f}")
    print(f"Default rate in test: {test_data['SeriousDlqin2yrs'].mean():.3f}")

if __name__ == "__main__":
    save_datasets()

