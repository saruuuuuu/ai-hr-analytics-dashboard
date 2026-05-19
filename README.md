🎯 Employee Performance Predictor using Data Analytics
An end-to-end machine learning system that predicts employee performance bands (High / Medium / Low) and surfaces actionable HR interventions using people analytics.

Python Streamlit scikit-learn License

📌 Problem Statement
Companies lose millions annually due to undetected performance issues, biased appraisals, and mis-targeted training budgets. Traditional HR review cycles are reactive — by the time low performance is identified, it's often too late for intervention.

This project builds a proactive, data-driven system that:

Predicts each employee's next performance band 1–2 months before appraisal
Identifies the top drivers of performance (both positive and negative)
Generates personalized L&D and coaching recommendations for HR teams
💼 Business Value
Use Case	Benefit
Early PIP flagging	Reduce appraisal surprises by 40–60%
Targeted L&D	Save training budget by directing courses to those who need them
Promotion decisions	Augment manager bias with data-backed evidence
Attrition prevention	Catch disengagement signals before resignation
Workforce planning	Forecast team performance for project staffing
🏗️ Architecture
Input (Employee Data)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  src/data_generator.py                              │
│  → 28 features: demographics, work signals,        │
│    engagement, attendance, feedback, HR signals     │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  src/train_model.py (Preprocessing Pipeline)        │
│  → Median Impute + Robust Scale (numerics)          │
│  → OHE (categoricals)                               │
│  → ColumnTransformer + Pipeline (no leakage)        │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  RandomForestClassifier                             │
│  → class_weight='balanced'                          │
│  → GridSearchCV (macro-F1, 5-fold stratified CV)   │
│  → Persisted with joblib                            │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  src/predictor.py                                   │
│  → Single prediction + probabilities                │
│  → Batch scoring (CSV output)                       │
│  → Recommendations Engine (10 intervention rules)   │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  app.py — Streamlit Dashboard                       │
│  → Overview, EDA, Live Predict, Batch Score,        │
│    HR Recommendations, Model Insights, Setup        │
└─────────────────────────────────────────────────────┘
🧰 Tech Stack
Layer	Tool
Language	Python 3.10+
Dashboard	Streamlit 1.32
ML	scikit-learn (RandomForest, LogisticRegression)
Data	Pandas, NumPy
Visualization	Plotly Express / Graph Objects
Model Persistence	joblib
Data Generation	NumPy random distributions
📁 Folder Structure
Employee-Performance-Predictor/
│
├── app.py                       ← Main Streamlit dashboard
├── requirements.txt
├── README.md
│
├── data/
│   └── employee_features.csv    ← 1000-row synthetic HR dataset
│
├── src/
│   ├── data_generator.py        ← Synthetic data with realistic correlations
│   ├── train_model.py           ← Full ML pipeline + GridSearch
│   └── predictor.py             ← Prediction utilities + recommendations
│
├── models/
│   ├── employee_perf_model.pkl  ← Trained RandomForest pipeline
│   └── label_encoder.pkl        ← Target label encoder
│
└── outputs/
    └── scored_employees.csv     ← Batch prediction results
🚀 How to Run
Prerequisites
Python 3.10+
pip
Installation
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Employee-Performance-Predictor.git
cd Employee-Performance-Predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
First-Time Setup (inside the app)
Open ⚙️ Train / Setup in the sidebar
Click Generate Dataset → creates 1000 synthetic employee records
Click Train Model → trains RandomForest with GridSearch (~1-2 min)
Navigate to any page to explore predictions, EDA, and recommendations
📊 Features
🏠 Overview
KPI dashboard: total employees, band distribution, avg scores
Department performance, training hours vs. score correlation
📊 Analytics & EDA
Interactive feature distributions by performance band
Full correlation matrix heatmap
Department, job level, gender segment analysis
🤖 Live Prediction
Manual entry OR lookup from dataset by employee ID
Probability chart for all 3 bands
Auto-generated HR recommendations based on shortfall analysis
📋 Batch Scoring
Score entire workforce in one click
Download CSV with predictions + probabilities
At-risk employee table (Low band)
🎯 HR Recommendations
Filtered by department, level, predicted band
Aggregated intervention frequency chart
Downloadable action items CSV
📈 Model Insights
Feature importance chart (MDI)
Fairness audit by gender, department, job level
High-band selection rates for bias detection
📐 Dataset Features (28 columns)
Category	Features
Demographics	age, gender, education, experience_years
Role	department, job_level, manager_tenure, salary
Work Signals	on_time_delivery_rate, avg_task_delay_days, projects_count, billable_hours_ratio
Quality	bug_count, code_review_score, qa_defect_density
Engagement	training_hours, certifications_count, hackathons_participated
Attendance	sick_days, unplanned_absences, avg_login_hours
Feedback	peer_feedback_score, manager_score, kudos_count
HR Signals	promotions_in_2y, salary_percentile_band
Target	perf_band_next (High / Medium / Low)
📈 Model Performance
Metric	Value
Algorithm	RandomForestClassifier
CV Strategy	Stratified 5-fold
Optimization	Macro-F1
Class Handling	class_weight='balanced'
Typical Accuracy	~85–90%
Typical Macro-F1	~0.84–0.88
🔮 Future Improvements
Real HR Data: Connect to HRMS APIs (Workday, BambooHR, SAP SuccessFactors)
Deep Learning: TabNet or transformer-based tabular models for complex interactions
Employee Attrition: Add parallel attrition prediction model
Real-time Scoring: FastAPI endpoint for HRMS integration
SHAP Explainability: Per-employee waterfall charts
Time Series: Predict performance trajectory over multiple quarters
MLOps: MLflow experiment tracking + scheduled monthly retraining
🎤 Interview Q&A
Q: Explain the project.

I built an ML system that classifies employees into High/Medium/Low performance bands using 28 features spanning work signals, engagement, feedback, and attendance. The model is a class-balanced RandomForest trained via macro-F1 optimized GridSearch. The Streamlit dashboard enables HR teams to do live predictions, batch scoring, and get actionable intervention recommendations.

Q: How did you handle class imbalance?

Used class_weight='balanced' in RandomForest, which automatically adjusts weights inversely proportional to class frequencies, combined with macro-F1 as the CV metric to prevent the model from ignoring minority classes.

Q: Why RandomForest over XGBoost?

RandomForest provides good out-of-box performance on tabular HR data, is robust to outliers via median imputation + robust scaling, and the MDI feature importances are directly interpretable for HR stakeholders. XGBoost would be the next step for marginal accuracy gains.

👨‍💻 Author
Built as a data science portfolio project demonstrating full-stack ML skills: data engineering → EDA → ML → explainability → deployment.

📄 License
MIT License — free to use and modify for educational and portfolio purposes.
