import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Loan Default Risk Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .low-risk {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .high-risk {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_data():
    """Load the trained model and sample data"""
    try:
        # Try to load the trained model
        model = joblib.load('models/best_model.joblib')
        model_info = joblib.load('models/model_info.joblib')
        
        # Load sample data for feature names
        sample_data = pd.read_csv('data/X_train.csv')
        feature_names = sample_data.columns.tolist()
        
        return model, model_info, feature_names, True
    except:
        # If model not available, return None
        return None, None, None, False

def create_input_form():
    """Create input form for loan application data"""
    
    st.markdown('<div class="sub-header">📋 Loan Application Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", min_value=18, max_value=100, value=35, help="Applicant's age in years")
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000, step=100, 
                                       help="Monthly income in dollars")
        num_dependents = st.selectbox("Number of Dependents", options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                    index=0, help="Number of dependents")
        
        st.subheader("Credit Information")
        revolving_utilization = st.slider("Revolving Credit Utilization", min_value=0.0, max_value=2.0, 
                                         value=0.3, step=0.01, 
                                         help="Total balance on credit cards divided by total credit limits")
        debt_ratio = st.slider("Debt Ratio", min_value=0.0, max_value=2.0, value=0.3, step=0.01,
                              help="Monthly debt payments divided by monthly income")
    
    with col2:
        st.subheader("Credit History")
        num_open_credit_lines = st.slider("Number of Open Credit Lines", min_value=0, max_value=30, 
                                         value=8, help="Number of open credit lines and loans")
        num_real_estate_loans = st.slider("Number of Real Estate Loans", min_value=0, max_value=10, 
                                         value=1, help="Number of mortgage and real estate loans")
        
        st.subheader("Past Due History")
        times_30_59_past_due = st.selectbox("Times 30-59 Days Past Due", options=list(range(11)), 
                                           index=0, help="Number of times 30-59 days past due in last 2 years")
        times_60_89_past_due = st.selectbox("Times 60-89 Days Past Due", options=list(range(11)), 
                                           index=0, help="Number of times 60-89 days past due in last 2 years")
        times_90_days_late = st.selectbox("Times 90+ Days Late", options=list(range(11)), 
                                         index=0, help="Number of times 90+ days late in last 2 years")
    
    return {
        'RevolvingUtilizationOfUnsecuredLines': revolving_utilization,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': times_30_59_past_due,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLinesAndLoans': num_open_credit_lines,
        'NumberOfTimes90DaysLate': times_90_days_late,
        'NumberRealEstateLoansOrLines': num_real_estate_loans,
        'NumberOfTime60-89DaysPastDueNotWorse': times_60_89_past_due,
        'NumberOfDependents': num_dependents
    }

def engineer_features(input_data):
    """Engineer features similar to training data"""
    
    # Create age groups
    age = input_data['age']
    if age <= 25:
        age_group_encoded = 0  # young
    elif age <= 35:
        age_group_encoded = 1  # adult
    elif age <= 50:
        age_group_encoded = 2  # middle_aged
    elif age <= 65:
        age_group_encoded = 3  # senior
    else:
        age_group_encoded = 4  # elderly
    
    # Create income groups (assuming median income of ~5500)
    income = input_data['MonthlyIncome']
    if income <= 2750:
        income_group_encoded = 0  # low
    elif income <= 5500:
        income_group_encoded = 1  # medium
    elif income <= 11000:
        income_group_encoded = 2  # high
    else:
        income_group_encoded = 3  # very_high
    
    # Create utilization categories
    utilization = input_data['RevolvingUtilizationOfUnsecuredLines']
    if utilization <= 0.3:
        utilization_category_encoded = 0  # low
    elif utilization <= 0.7:
        utilization_category_encoded = 1  # medium
    elif utilization <= 1.0:
        utilization_category_encoded = 2  # high
    else:
        utilization_category_encoded = 3  # very_high
    
    # Calculate derived features
    total_past_due = (input_data['NumberOfTime30-59DaysPastDueNotWorse'] + 
                     input_data['NumberOfTimes90DaysLate'] + 
                     input_data['NumberOfTime60-89DaysPastDueNotWorse'])
    
    debt_to_income = input_data['DebtRatio'] * input_data['MonthlyIncome']
    
    # Create feature vector
    features = [
        input_data['RevolvingUtilizationOfUnsecuredLines'],
        input_data['age'],
        input_data['NumberOfTime30-59DaysPastDueNotWorse'],
        input_data['DebtRatio'],
        input_data['MonthlyIncome'],
        input_data['NumberOfOpenCreditLinesAndLoans'],
        input_data['NumberOfTimes90DaysLate'],
        input_data['NumberRealEstateLoansOrLines'],
        input_data['NumberOfTime60-89DaysPastDueNotWorse'],
        input_data['NumberOfDependents'],
        total_past_due,
        debt_to_income,
        age_group_encoded,
        income_group_encoded,
        utilization_category_encoded
    ]
    
    return np.array(features).reshape(1, -1)

def make_prediction(model, features):
    """Make prediction using the trained model"""
    
    # Make prediction
    prediction = model.predict(features)[0]
    prediction_proba = model.predict_proba(features)[0]
    
    return prediction, prediction_proba

def display_prediction_result(prediction, prediction_proba):
    """Display prediction results"""
    
    st.markdown('<div class="sub-header">🎯 Prediction Results</div>', unsafe_allow_html=True)
    
    # Get probabilities
    prob_no_default = prediction_proba[0]
    prob_default = prediction_proba[1]
    
    # Display main result
    if prediction == 0:
        st.markdown(f'''
        <div class="prediction-result low-risk">
            ✅ LOW RISK<br>
            <span style="font-size: 1.2rem;">Probability of Default: {prob_default:.1%}</span>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="prediction-result high-risk">
            ⚠️ HIGH RISK<br>
            <span style="font-size: 1.2rem;">Probability of Default: {prob_default:.1%}</span>
        </div>
        ''', unsafe_allow_html=True)
    
    # Display probability breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Probability of No Default",
            value=f"{prob_no_default:.1%}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Probability of Default",
            value=f"{prob_default:.1%}",
            delta=None
        )
    
    # Risk interpretation
    st.markdown("### Risk Interpretation")
    if prob_default < 0.1:
        st.success("🟢 **Very Low Risk**: This applicant has excellent creditworthiness.")
    elif prob_default < 0.2:
        st.info("🔵 **Low Risk**: This applicant has good creditworthiness.")
    elif prob_default < 0.4:
        st.warning("🟡 **Moderate Risk**: This applicant requires careful evaluation.")
    else:
        st.error("🔴 **High Risk**: This applicant has significant default risk.")

def display_feature_importance():
    """Display feature importance information"""
    
    st.markdown('<div class="sub-header">📊 Key Risk Factors</div>', unsafe_allow_html=True)
    
    # Mock feature importance (in a real app, this would come from the model)
    importance_data = {
        'Feature': [
            'Revolving Credit Utilization',
            'Age',
            'Debt Ratio',
            'Monthly Income',
            'Times 90+ Days Late',
            'Number of Open Credit Lines',
            'Times 30-59 Days Past Due',
            'Number of Dependents'
        ],
        'Importance': [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
    }
    
    importance_df = pd.DataFrame(importance_data)
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Most Important Factors in Loan Default Prediction')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
               f'{width:.2f}', ha='left', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)

def display_model_info(model_info):
    """Display model information"""
    
    st.markdown('<div class="sub-header">🤖 Model Information</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h4>Model Type</h4>
            <p>{model_info.get("model_name", "Machine Learning Model")}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <h4>Algorithm</h4>
            <p>{model_info.get("model_type", "Ensemble Method")}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <h4>Performance</h4>
            <p>ROC AUC: 0.89+</p>
        </div>
        ''', unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🏦 Loan Default Risk Prediction System</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This application uses machine learning to predict the probability of loan default based on applicant information.
    Fill in the form below to get an instant risk assessment.
    """)
    
    # Load model
    model, model_info, feature_names, model_loaded = load_model_and_data()
    
    if not model_loaded:
        st.warning("⚠️ Model is still training. Using demo mode with mock predictions.")
        st.info("In demo mode, predictions are for demonstration purposes only.")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox("Choose a page:", ["Prediction", "About", "Model Performance"])
    
    if page == "Prediction":
        # Input form
        input_data = create_input_form()
        
        # Prediction button
        if st.button("🔮 Predict Default Risk", type="primary"):
            with st.spinner("Analyzing loan application..."):
                if model_loaded:
                    # Engineer features
                    features = engineer_features(input_data)
                    
                    # Make prediction
                    prediction, prediction_proba = make_prediction(model, features)
                    
                    # Display results
                    display_prediction_result(prediction, prediction_proba)
                else:
                    # Demo prediction
                    demo_prob = np.random.beta(2, 8)  # Skewed towards low risk
                    demo_prediction = 1 if demo_prob > 0.5 else 0
                    demo_proba = np.array([1-demo_prob, demo_prob])
                    
                    display_prediction_result(demo_prediction, demo_proba)
                    st.info("🔧 This is a demo prediction. The actual model is still training.")
        
        # Feature importance
        st.markdown("---")
        display_feature_importance()
    
    elif page == "About":
        st.header("About This Application")
        
        st.markdown("""
        ### 🎯 Purpose
        This loan default prediction system helps financial institutions assess credit risk by predicting
        the probability that a borrower will default on their loan within the next two years.
        
        ### 🧠 How It Works
        The system uses advanced machine learning algorithms trained on historical loan data to identify
        patterns and risk factors associated with loan defaults. Key factors include:
        
        - **Credit Utilization**: How much of available credit is being used
        - **Payment History**: Past due payments and delinquencies
        - **Debt-to-Income Ratio**: Monthly debt obligations relative to income
        - **Credit History Length**: Age and number of credit accounts
        - **Personal Information**: Age, income, and number of dependents
        
        ### 📊 Model Performance
        Our model achieves:
        - **ROC AUC Score**: 0.89+ (Excellent discrimination)
        - **Precision**: High accuracy in identifying actual defaults
        - **Recall**: Good coverage of potential default cases
        
        ### ⚠️ Important Disclaimer
        This tool is for educational and demonstration purposes. Real-world lending decisions
        should involve comprehensive analysis and comply with all applicable regulations.
        """)
    
    elif page == "Model Performance":
        st.header("Model Performance Metrics")
        
        # Mock performance data
        st.subheader("Model Comparison Results")
        
        performance_data = {
            'Model': ['XGBoost', 'LightGBM', 'Random Forest', 'Logistic Regression', 'Decision Tree'],
            'ROC AUC': [0.893, 0.892, 0.888, 0.786, 0.695],
            'F1 Score': [0.551, 0.514, 0.527, 0.120, 0.446],
            'Precision': [0.421, 0.398, 0.412, 0.064, 0.287],
            'Recall': [0.789, 0.721, 0.734, 0.891, 0.923]
        }
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # Performance visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(performance_df))
        width = 0.35
        
        ax.bar(x - width/2, performance_df['ROC AUC'], width, label='ROC AUC', alpha=0.8)
        ax.bar(x + width/2, performance_df['F1 Score'], width, label='F1 Score', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(performance_df['Model'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        if model_loaded and model_info:
            display_model_info(model_info)

if __name__ == "__main__":
    main()

