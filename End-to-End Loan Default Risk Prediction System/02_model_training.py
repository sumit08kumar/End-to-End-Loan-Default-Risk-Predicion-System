import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def load_processed_data():
    """Load preprocessed data"""
    
    print("=== LOADING PROCESSED DATA ===")
    
    X_train = pd.read_csv('data/X_train.csv')
    X_val = pd.read_csv('data/X_val.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()
    y_val = pd.read_csv('data/y_val.csv').values.ravel()
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Training target distribution: {np.bincount(y_train)}")
    print(f"Validation target distribution: {np.bincount(y_val)}")
    
    return X_train, X_val, X_test, y_train, y_val

def initialize_models():
    """Initialize all models to be compared"""
    
    print("\n=== INITIALIZING MODELS ===")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Neural Network': MLPClassifier(random_state=42, max_iter=500)
    }
    
    print(f"Initialized {len(models)} models for comparison")
    
    return models

def train_and_evaluate_models(models, X_train, X_val, y_train, y_val):
    """Train and evaluate all models"""
    
    print("\n=== TRAINING AND EVALUATING MODELS ===")
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred),
                'roc_auc': roc_auc_score(y_val, y_pred_proba)
            }
            
            results[name] = metrics
            trained_models[name] = model
            
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            
        except Exception as e:
            print(f"  Error training {name}: {str(e)}")
            continue
    
    return results, trained_models

def create_results_comparison(results):
    """Create comparison plots and tables"""
    
    print("\n=== CREATING RESULTS COMPARISON ===")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    
    print("Model Performance Comparison:")
    print(results_df.sort_values('roc_auc', ascending=False))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    metrics = ['accuracy', 'precision', 'recall', 'roc_auc']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        # Sort models by metric
        sorted_results = results_df.sort_values(metric, ascending=True)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(sorted_results)), sorted_results[metric])
        ax.set_yticks(range(len(sorted_results)))
        ax.set_yticklabels(sorted_results.index)
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('data/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results_df.to_csv('data/model_results.csv')
    
    return results_df

def hyperparameter_tuning(best_models, X_train, y_train):
    """Perform hyperparameter tuning for top models"""
    
    print("\n=== HYPERPARAMETER TUNING ===")
    
    # Define parameter grids for top models
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50]
        }
    }
    
    tuned_models = {}
    
    for model_name in best_models:
        if model_name in param_grids:
            print(f"\nTuning {model_name}...")
            
            # Get base model
            if model_name == 'Random Forest':
                base_model = RandomForestClassifier(random_state=42)
            elif model_name == 'XGBoost':
                base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            elif model_name == 'LightGBM':
                base_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model, 
                param_grids[model_name],
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"  Best parameters: {grid_search.best_params_}")
            print(f"  Best CV score: {grid_search.best_score_:.4f}")
            
            tuned_models[model_name] = grid_search.best_estimator_
    
    return tuned_models

def create_detailed_evaluation(best_model, model_name, X_val, y_val):
    """Create detailed evaluation for the best model"""
    
    print(f"\n=== DETAILED EVALUATION FOR {model_name} ===")
    
    # Make predictions
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    
    # Create evaluation plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Detailed Evaluation - {model_name}', fontsize=16)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
    axes[1, 0].plot(recall, precision)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    
    # 4. Feature Importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        feature_names = [f'Feature_{i}' for i in range(len(best_model.feature_importances_))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=True).tail(10)
        
        axes[1, 1].barh(range(len(importance_df)), importance_df['importance'])
        axes[1, 1].set_yticks(range(len(importance_df)))
        axes[1, 1].set_yticklabels(importance_df['feature'])
        axes[1, 1].set_xlabel('Feature Importance')
        axes[1, 1].set_title('Top 10 Feature Importances')
    
    plt.tight_layout()
    plt.savefig(f'data/{model_name.lower().replace(" ", "_")}_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_best_model(best_model, model_name, scaler=None):
    """Save the best model and preprocessing objects"""
    
    print(f"\n=== SAVING BEST MODEL: {model_name} ===")
    
    # Save model
    model_path = f'models/best_model_{model_name.lower().replace(" ", "_")}.joblib'
    joblib.dump(best_model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save model info
    model_info = {
        'model_name': model_name,
        'model_type': type(best_model).__name__,
        'model_path': model_path
    }
    
    joblib.dump(model_info, 'models/model_info.joblib')
    print("Model info saved")

def main():
    """Main training pipeline"""
    
    print("Starting Model Training Pipeline...")
    
    # Load processed data
    X_train, X_val, X_test, y_train, y_val = load_processed_data()
    
    # Initialize models
    models = initialize_models()
    
    # Train and evaluate models
    results, trained_models = train_and_evaluate_models(models, X_train, X_val, y_train, y_val)
    
    # Create results comparison
    results_df = create_results_comparison(results)
    
    # Get top 3 models
    top_models = results_df.sort_values('roc_auc', ascending=False).head(3).index.tolist()
    print(f"\nTop 3 models: {top_models}")
    
    # Hyperparameter tuning for top models
    tuned_models = hyperparameter_tuning(top_models, X_train, y_train)
    
    # Evaluate tuned models
    if tuned_models:
        print("\n=== EVALUATING TUNED MODELS ===")
        tuned_results, _ = train_and_evaluate_models(tuned_models, X_train, X_val, y_train, y_val)
        
        # Find best overall model
        all_results = {**results, **{f"{k}_tuned": v for k, v in tuned_results.items()}}
        best_model_name = max(all_results.keys(), key=lambda x: all_results[x]['roc_auc'])
        
        if '_tuned' in best_model_name:
            best_model = tuned_models[best_model_name.replace('_tuned', '')]
        else:
            best_model = trained_models[best_model_name]
    else:
        # Use best original model
        best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
        best_model = trained_models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best ROC AUC: {all_results[best_model_name]['roc_auc']:.4f}")
    
    # Create detailed evaluation
    create_detailed_evaluation(best_model, best_model_name, X_val, y_val)
    
    # Save best model
    save_best_model(best_model, best_model_name)
    
    print("\n=== MODEL TRAINING COMPLETED ===")
    print("Best model saved and ready for deployment!")
    
    return best_model, best_model_name, results_df

if __name__ == "__main__":
    best_model, best_model_name, results_df = main()

