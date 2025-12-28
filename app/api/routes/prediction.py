"""
    ML Prediction API Routes

    Exposes 4 ML endpoints:
    1. Payment time prediction
    2. Client risk scoring
    3. Client segmentation
    4. Revenue forecasting
"""
from fastapi import APIRouter, HTTPException, status
from app.schemas.requests import (
    PaymentPredictionRequest, PaymentPredictionResponse,
    RiskScoreRequest, RiskScoreResponse,
    ClientSegmentRequest, ClientSegmentResponse,
    RevenueForecastRequest, RevenueForecastResponse
)
from app.models.ml_models import ml_models
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# ============================================================================
# PAYMENT TIME PREDICTION
# ============================================================================
@router.post("/predict/payment-time", response_model=PaymentPredictionResponse)
async def predict_payment_time(request: PaymentPredictionRequest):
    """ Predict how many days until invoice payment """
    try:
        logger.info(f"Payment prediction request for amount: ${request.amount}")

        # model to pydantic dict
        features = request.model_dump()

        # call ML model
        result = ml_models.predict_payment_time(features)

        # Log result
        logger.info(
            f"Predicted {result['predicted_payment_days']} days "
            f"with {result['confidence_score']} confidence"
        )

        return PaymentPredictionResponse(**result)
    except Exception as e:
        # Log error
        logger.error(f"Payment prediction error: {str(e)}")
        
        # Return HTTP 500 error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
    
# ============================================================================
# CLIENT RISK SCORING
# ============================================================================
@router.post("/predict/risk-score", response_model=RiskScoreResponse)
async def calculate_risk_score(request: RiskScoreRequest):
    """ Calculate risk score for a client """
    try:
        logger.info(f"Risk scoring calculation request")

        # model to pydantic dict
        features = request.model_dump()

        # call ML model
        result = ml_models.calculate_risk_score(features)

        logger.info(f"Risk score: {result['risk_score']} ({result['risk_level']})")

        return RiskScoreResponse(**result)
    except Exception as e:
        # Log error
        logger.error(f"Risk scoring error: {str(e)}")
        
        # Return HTTP 500 error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk scoring failed: {str(e)}"
        )
# ============================================================================
# CLIENT SEGMENTATION
# ============================================================================
@router.post("/clients/segment", response_model=ClientSegmentResponse)
async def predict_client_segment(request: ClientSegmentRequest):
    """ Predict client behavioral segment """
    try:
        logger.info("Client segmentation request")
        
        # Convert request to dict
        features = request.model_dump()
        
        # Predict segment
        result = ml_models.predict_client_segment(features)
        
        logger.info(f"Segment: {result['segment_name']} (ID: {result['segment_id']})")
        
        return ClientSegmentResponse(**result)
        
    except Exception as e:
        logger.error(f"Segmentation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}"
        )

# ============================================================================
# REVENUE FORECASTING
# ============================================================================
@router.post("/forecast/revenue", response_model=RevenueForecastResponse)
async def forecast_revenue(request: RevenueForecastRequest):
    """ Forecast future monthly revenue """
    try:
        logger.info(f"Revenue forecast request for {request.months_ahead} months")
        
        # Convert invoice history to list of dicts
        invoice_history = [inv.model_dump() for inv in request.invoice_history]
        
        # Generate forecast
        result = ml_models.forecast_revenue(invoice_history, request.months_ahead)
        
        logger.info(f"Generated {len(result['forecasts'])} month forecast")
        
        return RevenueForecastResponse(**result)
        
    except Exception as e:
        logger.error(f"Revenue forecasting error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecasting failed: {str(e)}"
        )
