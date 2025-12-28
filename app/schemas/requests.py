"""
    Request and Response schemas for the Crefin ML API endpoints
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# ============================================================================
# PAYMENT TIME PREDICTION
# ============================================================================
class PaymentPredictionRequest(BaseModel):
    """Request model for payment time prediction"""

    # client features
    client_avg_payment_days: float = Field(..., description="Client's average payment days")
    client_late_payment_rate: float = Field(..., ge=0, le=1, description="Client's late payment rate (0-1)")
    client_payment_std: float = Field(..., ge=0, description="Client's payment standard deviation")
    client_total_invoices: int = Field(..., ge=0, description="Total invoices for this client")
    client_payment_trend: float = Field(..., description="Client's payment trend (positive = improving)")
    
    # invoice features
    amount: float = Field(..., gt=0, description="Invoice amount")
    issue_date: datetime = Field(..., description="Invoice issue date")

    class Config:
        json_schema_extra = {
            "example": {
                "client_avg_payment_days": 25.5,
                "client_late_payment_rate": 0.3,
                "client_payment_std": 5.2,
                "client_total_invoices": 15,
                "client_payment_trend": -2.5,
                "amount": 5000.00,
                "issue_date": "2025-12-21T00:00:00"
            }
        }

class PaymentPredictionResponse(BaseModel):
    """Response model for payment time prediction"""

    predicted_payment_days: float = Field(..., description="Predicted days until payment")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence (0-1)")
    predicted_payment_date: datetime = Field(..., description="Estimated payment date")
    feature_importance: Dict[str, float] = Field(..., description="Top contributing features")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_payment_days": 23.5,
                "confidence_score": 0.87,
                "predicted_payment_date": "2026-01-13T00:00:00",
                "feature_importance": {
                    "client_avg_payment_days": 0.35,
                    "client_late_payment_rate": 0.28,
                    "amount": 0.12
                }
            }
        }

# ============================================================================
# RISK SCORING
# ============================================================================
class RiskScoreRequest(BaseModel):
    """Request model for risk scoring"""

    client_avg_payment_days: float
    client_payment_std: float
    client_late_payment_rate: float = Field(..., ge=0, le=1)
    client_total_invoices: int = Field(..., ge=0)
    client_payment_trend: float
    days_since_last_invoice: int = Field(..., ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "client_avg_payment_days": 45.0,
                "client_payment_std": 12.5,
                "client_late_payment_rate": 0.75,
                "client_total_invoices": 20,
                "client_payment_trend": 5.0,
                "days_since_last_invoice": 30
            }
        }

class RiskScoreResponse(BaseModel):
    """Response model for risk scoring"""
    
    risk_score: int = Field(..., ge=0, le=100, description="Risk score (0-100)")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")
    risk_breakdown: Dict[str, float] = Field(..., description="Individual risk components")
    recommendation: str = Field(..., description="Recommended action")
    
    class Config:
        json_schema_extra = {
            "example": {
                "risk_score": 73,
                "risk_level": "high",
                "risk_breakdown": {
                    "late_payment_risk": 75.0,
                    "speed_risk": 70.0,
                    "consistency_risk": 65.0,
                    "trend_risk": 80.0,
                    "experience_risk": 25.0
                },
                "recommendation": "Require 50% upfront payment for new projects"
            }
        }

# ============================================================================
# CLIENT SEGMENTATION
# ============================================================================

class ClientSegmentRequest(BaseModel):
    """Request model for client segmentation"""
    
    client_avg_payment_days: float
    client_payment_std: float
    client_late_payment_rate: float = Field(..., ge=0, le=1)
    risk_score: int = Field(..., ge=0, le=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "client_avg_payment_days": 21.0,
                "client_payment_std": 3.5,
                "client_late_payment_rate": 0.15,
                "risk_score": 28
            }
        }


class ClientSegmentResponse(BaseModel):
    """Response model for client segmentation"""
    
    segment_id: int = Field(..., description="Segment ID (0-3)")
    segment_name: str = Field(..., description="Segment name")
    segment_characteristics: Dict[str, float] = Field(..., description="Segment averages")
    recommendation: str = Field(..., description="How to work with this segment")
    
    class Config:
        json_schema_extra = {
            "example": {
                "segment_id": 0,
                "segment_name": "Steady Medium",
                "segment_characteristics": {
                    "avg_payment_days": 21.0,
                    "avg_risk_score": 28.0,
                    "client_count": 18
                },
                "recommendation": "Standard Net-30 payment terms work well"
            }
        }


# ============================================================================
# REVENUE FORECASTING
# ============================================================================

class InvoiceHistoryItem(BaseModel):
    """Single invoice for revenue forecasting"""
    issue_date: datetime
    amount: float
    payment_days: int


class RevenueForecastRequest(BaseModel):
    """Request model for revenue forecasting"""
    
    invoice_history: List[InvoiceHistoryItem] = Field(..., min_length=3, description="Historical invoices (min 3)")
    months_ahead: int = Field(6, ge=1, le=12, description="Months to forecast (1-12)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "invoice_history": [
                    {"issue_date": "2025-01-15", "amount": 5000, "payment_days": 25},
                    {"issue_date": "2025-02-20", "amount": 6000, "payment_days": 30},
                    {"issue_date": "2025-03-10", "amount": 5500, "payment_days": 22}
                ],
                "months_ahead": 6
            }
        }


class MonthlyForecast(BaseModel):
    """Single month forecast"""
    month: str
    predicted_revenue: float
    lower_bound: float
    upper_bound: float
    confidence: int = Field(..., ge=0, le=100)


class RevenueForecastResponse(BaseModel):
    """Response model for revenue forecasting"""
    
    forecasts: List[MonthlyForecast]
    insights: Dict[str, Any] = Field(..., description="Business insights")
    
    class Config:
        json_schema_extra = {
            "example": {
                "forecasts": [
                    {
                        "month": "2026-01",
                        "predicted_revenue": 8500.0,
                        "lower_bound": 6200.0,
                        "upper_bound": 10800.0,
                        "confidence": 90
                    }
                ],
                "insights": {
                    "growth_rate": 15.5,
                    "trend": "Strong Growth",
                    "volatility": 25.3,
                    "best_month": {"date": "2025-11", "revenue": 9500}
                }
            }
        }
