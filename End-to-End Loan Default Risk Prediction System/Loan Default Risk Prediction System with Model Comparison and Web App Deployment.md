# Loan Default Risk Prediction System with Model Comparison and Web App Deployment

## Project Overview
This project aims to build a comprehensive machine learning pipeline to predict the probability of a customer defaulting on a loan. It involves data preprocessing, training and comparing multiple classification models, evaluating their performance, and deploying the best model as a web application.

## Problem Statement
Loan default prediction is a critical task for financial institutions to assess credit risk and make informed lending decisions. Accurate prediction helps in minimizing financial losses and optimizing loan portfolios.

## Dataset Description
The project uses a synthetic dataset based on the "Give Me Some Credit" competition structure, containing:
- **Training data**: 150,000 loan applications with 10 features
- **Test data**: 101,503 loan applications
- **Default rate**: ~10.4% (realistic imbalanced dataset)

### Key Features:
- RevolvingUtilizationOfUnsecuredLines: Credit utilization ratio
- Age: Applicant's age
- NumberOfTime30-59DaysPastDueNotWorse: Past due 30-59 days count
- DebtRatio: Monthly debt payments / monthly income
- MonthlyIncome: Monthly income in dollars
- NumberOfOpenCreditLinesAndLoans: Number of open credit lines
- NumberOfTimes90DaysLate: Past due 90+ days count
- NumberRealEstateLoansOrLines: Number of real estate loans
- NumberOfTime60-89DaysPastDueNotWorse: Past due 60-89 days count
- NumberOfDependents: Number of dependents

## Models Compared and Evaluation Metrics
We compared various supervised classification models:
* **Logistic Regression** - ROC AUC: 0.786
* **Decision Tree** - ROC AUC: 0.695
* **Random Forest** - ROC AUC: 0.888
* **XGBoost** - ROC AUC: 0.893 (Best Model)
* **LightGBM** - ROC AUC: 0.892

### Evaluation Metrics:
* **ROC AUC**: Area under the ROC curve
* **Precision-Recall Curve**: For imbalanced dataset evaluation
* **Confusion Matrix**: Classification accuracy breakdown
* **F1-Score**: Harmonic mean of precision and recall

## Web Application Features
The Streamlit web application provides:
- **Interactive Prediction Interface**: Input loan application details
- **Real-time Risk Assessment**: Instant default probability calculation
- **Risk Interpretation**: Color-coded risk levels with explanations
- **Feature Importance Visualization**: Key risk factors analysis
- **Model Performance Dashboard**: Comparison of different algorithms
- **About Page**: Detailed project information and disclaimers

## Web Application Screenshots
The deployed application includes:
1. **Prediction Interface**: User-friendly form for loan application data
2. **Results Display**: Clear risk assessment with probability scores
3. **Feature Importance**: Visual representation of key risk factors
4. **Model Performance**: Comparison charts and metrics

## Deployment Link
🚀 **Live Application**: https://8501-in6jd1ix1bzclipumbqti-d6dd4e01.manusvm.computer

## Project Structure
```
Loan-Default-Prediction/
│
├── data/                   # Raw and processed data
│   ├── cs-training.csv     # Training dataset
│   ├── cs-test.csv         # Test dataset
│   ├── X_train.csv         # Processed training features
│   ├── X_val.csv           # Processed validation features
│   ├── X_test.csv          # Processed test features
│   ├── y_train.csv         # Training labels
│   ├── y_val.csv           # Validation labels
│   └── model_results.csv   # Model comparison results
│
├── notebooks/              # EDA, modeling, experiments
│   ├── data_generation.py  # Synthetic data generation
│   ├── 01_eda_and_preprocessing.py  # EDA and preprocessing
│   └── 02_model_training_simple.py # Model training
│
├── models/                 # Saved ML models
│   ├── best_model.joblib   # Best trained model
│   └── model_info.joblib   # Model metadata
│
├── app/                    # Streamlit web application
│   └── streamlit_app.py    # Main application file
│
├── requirements_simple.txt # Python dependencies
├── Dockerfile              # Docker configuration
├── .dockerignore          # Docker ignore file
├── README.md              # Project documentation
└── todo.md                # Project progress tracking
```

## Instructions to Run Locally

### Prerequisites
- Python 3.11+
- pip package manager

### Setup and Installation
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Loan-Default-Prediction
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements_simple.txt
   ```

4. **Generate synthetic data:**
   ```bash
   cd notebooks
   python data_generation.py
   ```

5. **Run EDA and preprocessing:**
   ```bash
   python 01_eda_and_preprocessing.py
   ```

6. **Train models:**
   ```bash
   python 02_model_training_simple.py
   ```

7. **Launch the web application:**
   ```bash
   cd ../app
   streamlit run streamlit_app.py
   ```

8. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`

### Docker Deployment
1. **Build Docker image:**
   ```bash
   docker build -t loan-default-prediction .
   ```

2. **Run Docker container:**
   ```bash
   docker run -p 8501:8501 loan-default-prediction
   ```

## Key Technical Achievements

### Data Engineering
- ✅ Comprehensive EDA with visualizations
- ✅ Advanced feature engineering (age groups, income categories, derived features)
- ✅ Robust handling of missing values and outliers
- ✅ Proper train/validation/test splits with stratification

### Machine Learning
- ✅ Multiple algorithm comparison (5 different models)
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Proper evaluation metrics for imbalanced datasets
- ✅ Model persistence and reproducibility
- ✅ Feature importance analysis

### Web Application
- ✅ Professional Streamlit interface with custom CSS
- ✅ Interactive prediction form with input validation
- ✅ Real-time risk assessment and interpretation
- ✅ Comprehensive model performance dashboard
- ✅ Responsive design with multiple pages

### Deployment & DevOps
- ✅ Dockerized application for easy deployment
- ✅ Public web deployment with live URL
- ✅ Comprehensive documentation and README
- ✅ Clean project structure and code organization

## Model Performance Summary
The XGBoost model achieved the best performance:
- **ROC AUC**: 0.893 (Excellent discrimination)
- **F1 Score**: 0.551 (Good balance of precision/recall)
- **Precision**: 0.421 (42% of predicted defaults are actual defaults)
- **Recall**: 0.789 (79% of actual defaults are correctly identified)

## Important Disclaimer
This tool is for educational and demonstration purposes. Real-world lending decisions should involve comprehensive analysis, regulatory compliance, and human oversight. The synthetic dataset and model predictions should not be used for actual financial decisions.

## Future Enhancements
- [ ] Integration of SHAP/LIME for advanced explainability
- [ ] A/B testing framework for model comparison
- [ ] Real-time model monitoring and drift detection
- [ ] API endpoints for programmatic access
- [ ] Advanced feature engineering with external data sources
- [ ] Ensemble methods and stacking techniques

## License
This project is for educational purposes and demonstration of machine learning capabilities.


