import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_explore_data():
    """Load and perform initial exploration of the dataset"""
    
    print("=== LOADING DATA ===")
    # Load training data
    train_df = pd.read_csv('data/cs-training.csv')
    test_df = pd.read_csv('data/cs-test.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    print("\n=== BASIC INFO ===")
    print("Training data info:")
    print(train_df.info())
    
    print("\n=== TARGET VARIABLE DISTRIBUTION ===")
    target_counts = train_df['SeriousDlqin2yrs'].value_counts()
    print(target_counts)
    print(f"Default rate: {train_df['SeriousDlqin2yrs'].mean():.3f}")
    
    print("\n=== MISSING VALUES ===")
    missing_train = train_df.isnull().sum()
    missing_test = test_df.isnull().sum()
    
    print("Training data missing values:")
    print(missing_train[missing_train > 0])
    
    print("\nTest data missing values:")
    print(missing_test[missing_test > 0])
    
    print("\n=== BASIC STATISTICS ===")
    print(train_df.describe())
    
    return train_df, test_df

def create_visualizations(train_df):
    """Create EDA visualizations"""
    
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Loan Default Prediction - Exploratory Data Analysis', fontsize=16)
    
    # 1. Target distribution
    target_counts = train_df['SeriousDlqin2yrs'].value_counts()
    axes[0, 0].pie(target_counts.values, labels=['No Default', 'Default'], autopct='%1.1f%%')
    axes[0, 0].set_title('Target Variable Distribution')
    
    # 2. Age distribution by target
    train_df.boxplot(column='age', by='SeriousDlqin2yrs', ax=axes[0, 1])
    axes[0, 1].set_title('Age Distribution by Default Status')
    axes[0, 1].set_xlabel('Default Status')
    
    # 3. Monthly Income distribution (log scale)
    income_data = train_df['MonthlyIncome'].dropna()
    axes[1, 0].hist(np.log1p(income_data), bins=50, alpha=0.7)
    axes[1, 0].set_title('Monthly Income Distribution (Log Scale)')
    axes[1, 0].set_xlabel('Log(Monthly Income + 1)')
    
    # 4. Debt Ratio distribution
    debt_ratio_clean = train_df['DebtRatio'][train_df['DebtRatio'] < 2]  # Remove extreme outliers for visualization
    axes[1, 1].hist(debt_ratio_clean, bins=50, alpha=0.7)
    axes[1, 1].set_title('Debt Ratio Distribution (Outliers Removed)')
    axes[1, 1].set_xlabel('Debt Ratio')
    
    plt.tight_layout()
    plt.savefig('data/eda_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = train_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('data/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def handle_missing_values(df, is_training=True):
    """Handle missing values in the dataset"""
    
    print(f"\n=== HANDLING MISSING VALUES ({'Training' if is_training else 'Test'} Data) ===")
    
    df_processed = df.copy()
    
    # Handle MonthlyIncome missing values
    if 'MonthlyIncome' in df_processed.columns:
        median_income = df_processed['MonthlyIncome'].median()
        missing_income = df_processed['MonthlyIncome'].isnull().sum()
        print(f"Filling {missing_income} missing MonthlyIncome values with median: {median_income:.2f}")
        df_processed['MonthlyIncome'].fillna(median_income, inplace=True)
    
    # Handle NumberOfDependents missing values
    if 'NumberOfDependents' in df_processed.columns:
        mode_dependents = df_processed['NumberOfDependents'].mode()[0]
        missing_dependents = df_processed['NumberOfDependents'].isnull().sum()
        print(f"Filling {missing_dependents} missing NumberOfDependents values with mode: {mode_dependents}")
        df_processed['NumberOfDependents'].fillna(mode_dependents, inplace=True)
    
    return df_processed

def handle_outliers(df, is_training=True):
    """Handle outliers in the dataset"""
    
    print(f"\n=== HANDLING OUTLIERS ({'Training' if is_training else 'Test'} Data) ===")
    
    df_processed = df.copy()
    
    # Cap extreme values for RevolvingUtilizationOfUnsecuredLines
    col = 'RevolvingUtilizationOfUnsecuredLines'
    if col in df_processed.columns:
        q99 = df_processed[col].quantile(0.99)
        outliers = (df_processed[col] > q99).sum()
        print(f"Capping {outliers} outliers in {col} at 99th percentile: {q99:.3f}")
        df_processed[col] = np.clip(df_processed[col], 0, q99)
    
    # Cap extreme values for DebtRatio
    col = 'DebtRatio'
    if col in df_processed.columns:
        q99 = df_processed[col].quantile(0.99)
        outliers = (df_processed[col] > q99).sum()
        print(f"Capping {outliers} outliers in {col} at 99th percentile: {q99:.3f}")
        df_processed[col] = np.clip(df_processed[col], 0, q99)
    
    return df_processed

def feature_engineering(df, is_training=True):
    """Create new features"""
    
    print(f"\n=== FEATURE ENGINEERING ({'Training' if is_training else 'Test'} Data) ===")
    
    df_processed = df.copy()
    
    # Create age groups
    df_processed['age_group'] = pd.cut(df_processed['age'], 
                                     bins=[0, 25, 35, 50, 65, 100], 
                                     labels=['young', 'adult', 'middle_aged', 'senior', 'elderly'])
    
    # Create income groups (handle missing values first)
    income_median = df_processed['MonthlyIncome'].median()
    df_processed['income_group'] = pd.cut(df_processed['MonthlyIncome'], 
                                        bins=[0, income_median*0.5, income_median, income_median*2, float('inf')], 
                                        labels=['low', 'medium', 'high', 'very_high'])
    
    # Total past due count
    past_due_cols = ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 
                     'NumberOfTime60-89DaysPastDueNotWorse']
    df_processed['total_past_due'] = df_processed[past_due_cols].sum(axis=1)
    
    # Debt to income ratio
    df_processed['debt_to_income'] = df_processed['DebtRatio'] * df_processed['MonthlyIncome']
    
    # Credit utilization categories
    df_processed['utilization_category'] = pd.cut(df_processed['RevolvingUtilizationOfUnsecuredLines'],
                                                 bins=[0, 0.3, 0.7, 1.0, float('inf')],
                                                 labels=['low', 'medium', 'high', 'very_high'])
    
    print(f"Created {len(['age_group', 'income_group', 'total_past_due', 'debt_to_income', 'utilization_category'])} new features")
    
    return df_processed

def encode_categorical_features(train_df, test_df):
    """Encode categorical features"""
    
    print("\n=== ENCODING CATEGORICAL FEATURES ===")
    
    categorical_cols = ['age_group', 'income_group', 'utilization_category']
    
    # Use LabelEncoder for categorical features
    encoders = {}
    
    for col in categorical_cols:
        if col in train_df.columns:
            encoder = LabelEncoder()
            
            # Fit on training data
            train_df[col + '_encoded'] = encoder.fit_transform(train_df[col].astype(str))
            encoders[col] = encoder
            
            # Transform test data
            if col in test_df.columns:
                # Handle unseen categories in test data
                test_categories = test_df[col].astype(str)
                test_encoded = []
                
                for category in test_categories:
                    if category in encoder.classes_:
                        test_encoded.append(encoder.transform([category])[0])
                    else:
                        # Assign most frequent class for unseen categories
                        test_encoded.append(encoder.transform([encoder.classes_[0]])[0])
                
                test_df[col + '_encoded'] = test_encoded
    
    print(f"Encoded {len(categorical_cols)} categorical features")
    
    return train_df, test_df, encoders

def prepare_final_datasets(train_df, test_df):
    """Prepare final datasets for modeling"""
    
    print("\n=== PREPARING FINAL DATASETS ===")
    
    # Select features for modeling
    feature_cols = [
        'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
        'total_past_due', 'debt_to_income',
        'age_group_encoded', 'income_group_encoded', 'utilization_category_encoded'
    ]
    
    # Prepare training data
    X_train_full = train_df[feature_cols]
    y_train_full = train_df['SeriousDlqin2yrs']
    
    # Prepare test data
    X_test = test_df[feature_cols]
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    # Save processed datasets
    X_train_scaled.to_csv('data/X_train.csv', index=False)
    X_val_scaled.to_csv('data/X_val.csv', index=False)
    X_test_scaled.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_val.to_csv('data/y_val.csv', index=False)
    
    print("Processed datasets saved to data directory")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, scaler

def main():
    """Main preprocessing pipeline"""
    
    print("Starting EDA and Preprocessing Pipeline...")
    
    # Load and explore data
    train_df, test_df = load_and_explore_data()
    
    # Create visualizations
    create_visualizations(train_df)
    
    # Handle missing values
    train_df = handle_missing_values(train_df, is_training=True)
    test_df = handle_missing_values(test_df, is_training=False)
    
    # Handle outliers
    train_df = handle_outliers(train_df, is_training=True)
    test_df = handle_outliers(test_df, is_training=False)
    
    # Feature engineering
    train_df = feature_engineering(train_df, is_training=True)
    test_df = feature_engineering(test_df, is_training=False)
    
    # Encode categorical features
    train_df, test_df, encoders = encode_categorical_features(train_df, test_df)
    
    # Prepare final datasets
    X_train, X_val, X_test, y_train, y_val, scaler = prepare_final_datasets(train_df, test_df)
    
    print("\n=== PREPROCESSING COMPLETED ===")
    print("Ready for model training!")
    
    return X_train, X_val, X_test, y_train, y_val, scaler, encoders

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, scaler, encoders = main()

