"""
    Main FastAPI application for Crefin ML API

    This file creates the FastAPI app, configures middleware,
    loads ML models, and sets up all routes.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import prediction
from app.models.ml_models import ml_models
import logging

# ============================================================================
# CONFIGURE LOGGING
# ============================================================================
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CREATE FASTAPI APP
# ============================================================================
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# STARTUP EVENT - LOAD ML MODELS
# ============================================================================
@app.on_event("startup")
async def startup_event():
    """ Runs on startup and Loads all ML models into memory """
    logger.info(f"🚀 Starting {settings.API_TITLE} ({settings.ENVIRONMENT})...")
    logger.info(f"📊 API Version: {settings.API_VERSION}")
    logger.info(f"🌐 CORS Origins: {settings.cors_origins_list}")

    try:
        # load all models
        ml_models.load_models(settings)
        logger.info("✅ ML models loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load ML models: {str(e)}")
        raise 

# ============================================================================
# SHUTDOWN EVENT
# ============================================================================
@app.on_event("shutdown")
async def shutdown_event():
    """ Runs once server shutsdown, clean up resources if needed """
    logger.info("🛑 Shutting down Crefin ML API...")

# ============================================================================
# INCLUDE API ROUTES
# ============================================================================
app.include_router(
    prediction.router,
    prefix="/api/ml",
    tags=["ML Predictions"]
)

# ============================================================================
# ROOT ENDPOINT
# ============================================================================
@app.get("/")
async def root():
    """ Root endpoint - API information
        Returns the service info and available endpoints.
    """
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "running",
        "endpoints": {
            "payment_prediction": "/api/ml/predict/payment-time",
            "risk_scoring": "/api/ml/clients/risk-score",
            "client_segmentation": "/api/ml/clients/segment",
            "revenue_forecasting": "/api/ml/forecast/revenue"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    }

# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/health")
async def health_check():
    """
        Health check endpoint
        
        Returns API health status and whether models are loaded
        Used by monitoring systems and load balancers
    """
    # Check if all models are loaded
    models_loaded = all([
        ml_models.payment_predictor is not None,
        ml_models.risk_scorer is not None,
        ml_models.client_segmenter is not None,
        ml_models.revenue_forecaster is not None
    ])
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "environment": settings.ENVIRONMENT,
        "version": settings.API_VERSION
    }