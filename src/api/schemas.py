"""
schemas.py - Pydantic Models for API Request/Response
======================================================
Strong typing with validation — shows production engineering mindset.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum



class LoanGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class HomeOwnership(str, Enum):
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"


class LoanApplication(BaseModel):
    """Input schema for a single loan application."""
    
    loan_amnt: float = Field(..., gt=0, le=100000, description="Loan amount in USD")
    funded_amnt: Optional[float] = Field(default=None, ge=0, le=100000, description="Funded amount (defaults to loan_amnt)")
    annual_inc: float = Field(..., gt=0, description="Annual income in USD")
    dti: float = Field(..., ge=0, le=100, description="Debt-to-income ratio")
    int_rate: float = Field(..., ge=0, le=1, description="Interest rate (decimal)")
    installment: float = Field(..., gt=0, description="Monthly payment in USD")
    grade: LoanGrade = Field(..., description="Loan grade A-G")
    term_months: int = Field(default=36, description="Loan term in months (36 or 60)")
    revol_util: float = Field(default=0, ge=0, le=200, description="Revolving utilization %")
    revol_bal: float = Field(default=0, ge=0, description="Revolving balance")
    open_acc: int = Field(default=5, ge=0, description="Open accounts")
    total_acc: int = Field(default=10, ge=0, description="Total accounts")
    pub_rec: int = Field(default=0, ge=0, description="Public records")
    delinq_2yrs: int = Field(default=0, ge=0, description="Delinquencies in 2 years")
    inq_last_6mths: int = Field(default=0, ge=0, description="Inquiries last 6 months")
    fico_score: float = Field(default=700, ge=300, le=850, description="FICO score")
    emp_length_years: float = Field(default=5, ge=0, le=50, description="Employment years")
    
    @field_validator("annual_inc")
    @classmethod
    def income_must_be_reasonable(cls, v):
        if v > 10_000_000:
            raise ValueError("Annual income exceeds $10M — please verify")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt": 15000, "annual_inc": 65000, "dti": 18.5,
                "int_rate": 0.12, "installment": 450, "grade": "C",
                "revol_util": 55.3, "fico_score": 680,
            }
        }


class RiskPrediction(BaseModel):
    """Output schema for a risk prediction."""
    
    applicant_id: str
    default_probability: float = Field(..., ge=0, le=1)
    risk_segment: str
    recommendation: str  # APPROVE, REVIEW, REJECT
    confidence: float
    top_risk_factors: List[str]
    cached: bool = False


class PolicyRequest(BaseModel):
    """Input for policy simulation."""
    
    threshold: float = Field(default=0.35, ge=0.05, le=0.95)
    max_default_rate: float = Field(default=0.10, ge=0.01, le=0.50)
    min_acceptance_rate: float = Field(default=0.40, ge=0.10, le=0.95)


class PolicyResult(BaseModel):
    """Output for policy simulation."""
    
    threshold: float
    acceptance_rate_pct: float
    default_rate_pct: float
    default_reduction_pct: float
    approved_count: int
    expected_defaults: int
    is_optimal: bool


class HealthResponse(BaseModel):
    """API health check response."""
    
    status: str = "healthy"
    model_loaded: bool = False
    cache_size: int = 0
    cache_hit_rate: float = 0.0
    model_loaded: bool
    cache_size: int
    cache_hit_rate: float
    version: str = "2.0.0"
