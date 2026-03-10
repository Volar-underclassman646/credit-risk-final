"""
test_api.py - Tests for the FastAPI scoring endpoint
=====================================================
Run: pytest tests/test_api.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.schemas import LoanApplication, RiskPrediction, LoanGrade


class TestLoanApplicationValidation:
    """Test Pydantic input validation."""
    
    def test_valid_application(self):
        app = LoanApplication(
            loan_amnt=15000, annual_inc=65000, dti=18.5,
            int_rate=0.12, installment=450, grade=LoanGrade.C
        )
        assert app.loan_amnt == 15000
    
    def test_negative_loan_rejected(self):
        with pytest.raises(Exception):
            LoanApplication(loan_amnt=-5000, annual_inc=65000, dti=18.5,
                          int_rate=0.12, installment=450, grade=LoanGrade.C)
    
    def test_extreme_income_rejected(self):
        with pytest.raises(Exception):
            LoanApplication(loan_amnt=15000, annual_inc=999999999, dti=18.5,
                          int_rate=0.12, installment=450, grade=LoanGrade.C)
    
    def test_invalid_grade_rejected(self):
        with pytest.raises(Exception):
            LoanApplication(loan_amnt=15000, annual_inc=65000, dti=18.5,
                          int_rate=0.12, installment=450, grade="Z")
    
    def test_dti_boundary(self):
        app = LoanApplication(loan_amnt=15000, annual_inc=65000, dti=0,
                            int_rate=0.12, installment=450, grade=LoanGrade.A)
        assert app.dti == 0
    
    def test_defaults_applied(self):
        app = LoanApplication(loan_amnt=15000, annual_inc=65000, dti=18.5,
                            int_rate=0.12, installment=450, grade=LoanGrade.C)
        assert app.fico_score == 700  # Default
        assert app.pub_rec == 0       # Default


class TestRiskPredictionSchema:
    """Test output schema."""
    
    def test_valid_prediction(self):
        pred = RiskPrediction(
            applicant_id="abc123", default_probability=0.25,
            risk_segment="Medium Risk", recommendation="REVIEW",
            confidence=0.75, top_risk_factors=["High DTI"]
        )
        assert pred.recommendation == "REVIEW"
    
    def test_probability_bounds(self):
        with pytest.raises(Exception):
            RiskPrediction(
                applicant_id="abc", default_probability=1.5,
                risk_segment="High", recommendation="REJECT",
                confidence=0.5, top_risk_factors=[]
            )
