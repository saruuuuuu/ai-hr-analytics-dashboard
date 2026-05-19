"""
src/data_generator.py
---------------------
Generates a realistic synthetic HR dataset for Employee Performance Prediction.
Simulates real company patterns: correlations between features and performance.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

np.random.seed(42)

DEPARTMENTS = ["Engineering", "Sales", "HR", "Finance", "Marketing", "Operations", "Product", "Data Science"]
EDUCATION = ["Bachelor's", "Master's", "PhD", "Diploma"]
JOB_LEVELS = ["Junior", "Mid", "Senior", "Lead", "Manager"]
GENDERS = ["Male", "Female", "Non-binary"]

def generate_employee_dataset(n_employees: int = 1000, save_path: str = None) -> pd.DataFrame:
    """
    Generate n_employees rows of realistic synthetic HR data.
    Performance labels are derived from weighted signal combinations
    to mimic real-world patterns.
    """

    # --- Base demographics ---
    ages = np.random.normal(35, 8, n_employees).clip(22, 60).astype(int)
    experience = (ages - 22 + np.random.randint(-3, 4, n_employees)).clip(0, 38).astype(int)
    gender = np.random.choice(GENDERS, n_employees, p=[0.52, 0.44, 0.04])
    education = np.random.choice(EDUCATION, n_employees, p=[0.45, 0.35, 0.10, 0.10])
    department = np.random.choice(DEPARTMENTS, n_employees)
    job_level = np.random.choice(JOB_LEVELS, n_employees, p=[0.25, 0.30, 0.25, 0.12, 0.08])
    manager_tenure = np.random.randint(1, 10, n_employees)

    # --- Salary (correlated with experience & job level) ---
    level_mult = {"Junior": 1.0, "Mid": 1.4, "Senior": 1.8, "Lead": 2.2, "Manager": 2.8}
    base_salary = 400000
    salary = np.array([
        int(base_salary * level_mult[job_level[i]] * (1 + 0.03 * experience[i]) * np.random.uniform(0.85, 1.15))
        for i in range(n_employees)
    ])

    # --- Work signals (0-1 scale for delivery, others raw counts) ---
    on_time_delivery_rate = np.random.beta(6, 2, n_employees).round(2)         # skewed high
    avg_task_delay_days   = np.random.exponential(3, n_employees).clip(0, 30).round(1)
    projects_count        = np.random.poisson(4, n_employees).clip(1, 12)
    billable_hours_ratio  = np.random.beta(5, 2, n_employees).round(2)

    # --- Quality signals ---
    bug_count             = (np.random.poisson(5, n_employees) * (1 - on_time_delivery_rate * 0.4)).clip(0, 30).astype(int)
    code_review_score     = np.random.beta(4, 2, n_employees).round(2) * 5    # 0-5
    qa_defect_density     = np.random.exponential(2, n_employees).clip(0, 15).round(2)

    # --- Engagement & learning ---
    training_hours        = np.random.gamma(3, 8, n_employees).clip(0, 100).astype(int)
    certifications_count  = np.random.poisson(1.5, n_employees).clip(0, 8)
    hackathons            = np.random.poisson(0.8, n_employees).clip(0, 5)

    # --- Attendance ---
    sick_days             = np.random.poisson(4, n_employees).clip(0, 30).astype(int)
    unplanned_absences    = np.random.poisson(2, n_employees).clip(0, 15).astype(int)
    avg_login_hours       = np.random.normal(7.5, 1.5, n_employees).clip(4, 12).round(1)

    # --- Feedback scores ---
    peer_feedback_score   = np.random.beta(4, 2, n_employees).round(2) * 5     # 0-5
    manager_score_prev    = np.random.beta(4, 2, n_employees).round(2) * 5     # 0-5
    kudos_count           = np.random.poisson(5, n_employees).clip(0, 30)

    # --- HR signals ---
    promotions_in_2y      = np.random.choice([0, 1, 2], n_employees, p=[0.65, 0.28, 0.07])
    salary_percentile     = np.random.choice(["Low", "Mid", "High"], n_employees, p=[0.30, 0.45, 0.25])

    # ----------------------------------------------------------------
    # Derive performance score (0-100) from weighted signals
    # This ensures realistic label distribution and feature correlations
    # ----------------------------------------------------------------
    score = (
        0.20 * (on_time_delivery_rate * 100) +
        0.15 * (manager_score_prev / 5 * 100) +
        0.12 * (peer_feedback_score / 5 * 100) +
        0.10 * (billable_hours_ratio * 100) +
        0.08 * (training_hours / 100 * 100).clip(0, 100) +
        0.08 * (code_review_score / 5 * 100) +
        0.07 * (certifications_count / 8 * 100) +
        0.06 * np.clip(100 - avg_task_delay_days * 5, 0, 100) +
        0.05 * (kudos_count / 30 * 100) +
        0.05 * np.clip(100 - bug_count * 4, 0, 100) +
        0.04 * np.clip(100 - sick_days * 3, 0, 100)
    )
    score = score + np.random.normal(0, 5, n_employees)   # add noise
    score = score.clip(0, 100)

    # Convert to 3-class label using quantile-based thresholds for balanced classes
    low_thr  = np.percentile(score, 33)   # bottom 33% → Low
    high_thr = np.percentile(score, 67)   # top 33%    → High
    perf_band = pd.cut(score, bins=[-np.inf, low_thr, high_thr, np.inf],
                       labels=["Low", "Medium", "High"])

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "employee_id": [f"EMP{str(i+1001).zfill(5)}" for i in range(n_employees)],
        "age": ages,
        "gender": gender,
        "education": education,
        "experience_years": experience,
        "department": department,
        "job_level": job_level,
        "manager_tenure": manager_tenure,
        "salary": salary,
        "on_time_delivery_rate": on_time_delivery_rate,
        "avg_task_delay_days": avg_task_delay_days,
        "projects_count": projects_count,
        "billable_hours_ratio": billable_hours_ratio,
        "bug_count": bug_count,
        "code_review_score": code_review_score.round(2),
        "qa_defect_density": qa_defect_density,
        "training_hours": training_hours,
        "certifications_count": certifications_count,
        "hackathons_participated": hackathons,
        "sick_days": sick_days,
        "unplanned_absences": unplanned_absences,
        "avg_login_hours": avg_login_hours,
        "peer_feedback_score": peer_feedback_score.round(2),
        "manager_score": manager_score_prev.round(2),
        "kudos_count": kudos_count,
        "promotions_in_2y": promotions_in_2y,
        "salary_percentile_band": salary_percentile,
        "performance_score": score.round(1),
        "perf_band_next": perf_band,
    })

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✅ Dataset saved → {save_path}  ({len(df)} rows)")

    return df


if __name__ == "__main__":
    df = generate_employee_dataset(1000, save_path="../data/employee_features.csv")
    print(df.head())
    print("\nClass distribution:\n", df["perf_band_next"].value_counts())
