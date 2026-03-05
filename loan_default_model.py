import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# ============================================================
# PART 3
# Loan Default Probability Model
# ============================================================

# Load the borrower dataset
data = pd.read_csv("loan_data.csv")

# ------------------------------------------------------------
# Create additional financial ratios
# ------------------------------------------------------------

# Debt-to-income ratio
data["debt_to_income"] = data["total_debt_outstanding"] / data["income"]

# Payment-to-income ratio
data["payment_to_income"] = data["loan_amt_outstanding"] / data["income"]

# ------------------------------------------------------------
# Define model features
# ------------------------------------------------------------

features = [
    "credit_lines_outstanding",
    "debt_to_income",
    "payment_to_income",
    "years_employed",
    "fico_score"
]

X = data[features]
y = data["default"]

# ------------------------------------------------------------
# Train logistic regression model
# ------------------------------------------------------------

model = LogisticRegression(max_iter=10000)

model.fit(X, y)

# ============================================================
# Function to estimate probability of default
# ============================================================

def predict_pd(
    credit_lines_outstanding,
    debt_to_income,
    payment_to_income,
    years_employed,
    fico_score
):

    borrower = np.array([[
        credit_lines_outstanding,
        debt_to_income,
        payment_to_income,
        years_employed,
        fico_score
    ]])

    pd_value = model.predict_proba(borrower)[0][1]

    return pd_value


# ============================================================
# Expected loss calculation
# ============================================================

def expected_loss(
    loan_amount,
    credit_lines_outstanding,
    debt_to_income,
    payment_to_income,
    years_employed,
    fico_score
):

    PD = predict_pd(
        credit_lines_outstanding,
        debt_to_income,
        payment_to_income,
        years_employed,
        fico_score
    )

    recovery_rate = 0.10
    LGD = 1 - recovery_rate

    loss = PD * LGD * loan_amount

    return loss


# ============================================================
# Example test borrower
# ============================================================

test_loss = expected_loss(
    loan_amount=5000,
    credit_lines_outstanding=2,
    debt_to_income=0.35,
    payment_to_income=0.15,
    years_employed=5,
    fico_score=650
)

print("Expected Loss:", test_loss)
