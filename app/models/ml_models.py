"""
    ML Model loader and wrapper
    Loads saved models and provides prediction interfaces
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging
import sys

from app.models.risk_scorer import ClientRiskScorer
from app.models.client_segmenter import ClientSegmenter
from app.models.revenue_forecaster import RevenueForecaster

logger = logging.getLogger(__name__)

class MLModels:
    """Singleton class to load and manage ML models"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLModels, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.payment_predictor = None
            self.payment_predictor_metadata = None
            self.risk_scorer = None
            self.client_segmenter = None
            self.revenue_forecaster = None
            self._initialized = True

    def load_models(self, config):
        """ Load all ML modes from disk """

        logger.info("Loading ML models...")

        sys.modules['models'] = sys.modules['app.models']
        sys.modules['models.risk_scorer'] = sys.modules['app.models.risk_scorer']
        sys.modules['models.client_segmenter'] = sys.modules['app.models.client_segmenter']
        sys.modules['models.revenue_forecaster'] = sys.modules['app.models.revenue_forecaster']

        try:
            # Load Payment Predictor
            logger.info(f"Loading payment predictor from {config.PAYMENT_PREDICTOR_PATH}")
            self.payment_predictor = joblib.load(config.PAYMENT_PREDICTOR_PATH)
            self.payment_predictor_metadata = joblib.load(config.PAYMENT_PREDICTOR_METADATA_PATH)
            logger.info(f"✅ Payment predictor loaded (R² = {self.payment_predictor_metadata.get('r2', 'N/A')})")
            
            # load risk scorer
            logger.info(f"Loading risk scorer from {config.RISK_SCORER_PATH}")
            self.risk_scorer = joblib.load(config.RISK_SCORER_PATH)
            logger.info("✅ Risk scorer loaded")
            
            # Load Client Segmenter
            logger.info(f"Loading client segmenter from {config.CLIENT_SEGMENTER_PATH}")
            self.client_segmenter = joblib.load(config.CLIENT_SEGMENTER_PATH)
            logger.info("✅ Client segmenter loaded")
            
            # Load Revenue Forecaster
            logger.info(f"Loading revenue forecaster from {config.REVENUE_FORECASTER_PATH}")
            self.revenue_forecaster = joblib.load(config.REVENUE_FORECASTER_PATH)
            logger.info("✅ Revenue forecaster loaded")
            
            logger.info("🎉 All ML models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            raise

    def predict_payment_time(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """ Predict payment time for an invoice """

        if self.payment_predictor is None:
            raise RuntimeError("Payment predictor model is not loaded")
        
        # Get feature names from metadata
        feature_names = self.payment_predictor_metadata['feature_names']

        # engineer features
        engineered_features = self._engineer_payment_features(features, feature_names)

        # Create feature array in correct order
        X = np.array([[engineered_features[f] for f in feature_names]])

        # get prediction days
        predicted_days = float(self.payment_predictor.predict(X)[0])

        # calculate predicted payment date
        issue_date = features['issue_date']
        predicted_payment_date = issue_date + timedelta(days=int(predicted_days))

        # Get feature importance
        feature_importances = dict(zip(
            feature_names,
            self.payment_predictor.feature_importances_
        ))

        # Get top 5 features
        top_features = dict(sorted(
            feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])

        # Get confidence score
        confidence_score = float(self.payment_predictor_metadata['r2'])

        return {
            "predicted_payment_days": round(predicted_days, 1),
            "confidence_score": round(confidence_score, 2),
            "predicted_payment_date": predicted_payment_date,
            "feature_importance": {k: round(v, 3) for k, v in top_features.items()}
        }
    
    def _engineer_payment_features(self, features: Dict, feature_names: List[str]) -> Dict[str, float]:
        """ 
            Engineer features for payment prediction
            Takes user's input of 7 fields and creates model's complex input 18 fields

            Args:
                features: Dict - user input features
                feature_names: List[str] - list of features the model expects

            Returns:
                Dictionary with all 18 features
        
        """

        engineered = {}

        # copy the basic client features (provided by the user)
        engineered['client_avg_payment_days'] = features['client_avg_payment_days']
        engineered['client_late_payment_rate'] = features['client_late_payment_rate']
        engineered['client_payment_std'] = features['client_payment_std']
        engineered['client_total_invoices'] = features['client_total_invoices']
        engineered['client_payment_trend'] = features['client_payment_trend']

        # copy invoice features
        engineered['amount'] = features['amount']

        # extract date-based features from issue_date
        issue_date = features['issue_date']

        # extract the month number
        engineered['month'] = issue_date.month

        # create bool features
        engineered['is_quarter_end'] = 1 if issue_date.month in [3, 6, 9, 12] else 0
        # some companies delay payments during quarter_end

        engineered['is_month_end'] = 1 if issue_date.day >= 25 else 0
        # delay payment, probably accounting team is busy closing month accounts

        engineered['is_holiday_season'] = 1 if issue_date.month in [11, 12] else 0
        # vacation time, slower payments

        # calculating derived features
        # how this invoice amount compares to client's avg payment time?
        # relative measure
        # example: amount = 5000, client_avg_payment_days = 25
        # engineered['amount_vs_client_avg'] = 5000 / (25 * 100) = 2.0
        # meaning this invoice is 2x the average daily payment amount for this client
        # if result > 1: larger invoice than usual -> might take longer, vice versa is true
        if features['client_avg_payment_days'] > 0:
            engineered['amount_vs_client_avg'] = (
                features['amount'] / (features['client_avg_payment_days'] * 100)
            )
        else:
            engineered['amount_vs_client_avg'] = 0.0

        # fill any missing features if any exists
        for fname in feature_names:
            if fname not in engineered:
                engineered[fname] = 0.0

        return engineered
    
    def calculate_risk_score(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """ 
            Calculate client risk score using the risk scorer model

            Args:
                features: Dict - with risk_score, risk_level, breakdown, recommendation
        """

        # check if model is loaded
        if self.risk_scorer is None:
            raise RuntimeError("Risk scorer model is not loaded")
        
        # since the risk_scorer model already has a risk calculation method,
        # we just call it directly
        result = self.risk_scorer.calculate_risk_score(features)

        return result
    
    def predict_client_segment(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """ 
            Predict which sengment a client belongs to

            Uses K-Means clustring to group clients into 4 behavioral segments

            Args:
                features: Client features (avg_payment_days, std, late_rate, risk_score)

            Returns:
                Dict with segment_id, segment_name, characteristics, recommendation
        """
        # check if model is loaded
        if self.client_segmenter is None:
            raise RuntimeError("Client segmenter model is not loaded")
        
        # K-Means expects features in a specific order
        # create 2D-array: [[feature1, feature2, feature3, feature4]]

        X = np.array([[
            features['client_avg_payment_days'],
            features['client_payment_std'],
            features['client_late_payment_rate'],
            features['risk_score']
        ]])

        # scale features
        # bse, if not scaled, large values dominate the distance calculations for K-Means
        X_scaled = self.client_segmenter.scaler.transform(X)
        # scaler was already FIT during training so we use transform instead of fit_transform

        # predict segment
        segment_id = int(self.client_segmenter.kmeans.predict(X_scaled)[0])
        # How K-Means works:
        # 1. Calculate distance to each cluster center
        # 2. Return the closest cluster
        #
        # Example:
        # Distance to cluster 0: 0.5
        # Distance to cluster 1: 2.3
        # Distance to cluster 2: 0.8
        # Distance to cluster 3: 1.9
        # → Closest is cluster 0 → segment_id = 0

        # get segment profile
        segment_profile = self.client_segmenter.segment_profiles.get(segment_id, {})
        # Example segment_profile:
        # {
        #     'name': 'Steady Medium',
        #     'characteristics': {
        #         'avg_payment_days': 21.0,
        #         'avg_risk_score': 28.0,
        #         'client_count': 18
        #     },
        #     'recommendation': 'Standard Net-30 payment terms work well'
        # }

        # format and return the result
        return {
            "segment_id": segment_id,
            "segment_name": segment_profile.get('name', 'Unknown'),
            "segment_characteristics": segment_profile.get('characteristics', {}),
            "recommendation": segment_profile.get('recommendation', 'Standard payment terms')
        }
    
    def forecast_revenue(self, invoice_history: List[Dict], months_ahead: int = 6) -> Dict[str, Any]:
        """ 
        Forecast future revenue based on historical invoice data.

        Args:
            invoice_history: List[Dict] - List of past invoices with relevant features
            months_ahead: int - Number of months to forecast (default 6)

        Returns:
            Dict[str, Any] - Forecasted revenue figures
        """
        # check if model is loaded
        if self.revenue_forecaster is None:
            raise RuntimeError("Revenue forecaster model is not loaded")
        
        # convert invoice history to df
        # forecaster expects a pandas DataFrame
        invoices_df = pd.DataFrame([
            {
                'issue_date': pd.to_datetime(inv['issue_date']),
                'amount': inv['amount'],
                'payment_days': inv['payment_days']
            }
            for inv in invoice_history
        ])

        # train forecaster on historical data
        # forecaster trains everytime bse revenue patterns change over time
        # each user has their own unique revenue patterns, so we train on their specific history
        self.revenue_forecaster.fit(invoices_df)
        # What happens inside .fit():
        # 1. Groups invoices by month
        # 2. Calculates monthly revenue totals
        # 3. Fits linear regression to find trend
        # 4. Stores monthly statistics
        # 5. Calculates growth rate

        # generate forecast
        forecast_df = self.revenue_forecaster.forecast(months_ahead=months_ahead)

        # What happens inside .forecast():
        # 1. Projects trend line into future
        # 2. Adjusts for seasonality (if detected)
        # 3. Calculates confidence intervals
        # 4. Returns predictions with ranges
        #
        # Example output:
        #     month  predicted_revenue  lower_bound  upper_bound  confidence
        # 0  2026-01             8500         6200        10800          90
        # 1  2026-02             9200         6800        11600          85
        # 2  2026-03             9800         7200        12400          80

        # get business insights
        insights = self.revenue_forecaster.get_insights()

        # What's in insights:
        # {
        #     'current_month_revenue': 10048.75,
        #     'avg_monthly_revenue': 6613.33,
        #     'growth_rate': 71.8,
        #     'trend': 'Strong Growth',
        #     'volatility': 40.6,
        #     'consistency': 'Moderately Stable',
        #     'best_month': {'date': '2025-11', 'revenue': 10930.25},
        #     'worst_month': {'date': '2025-01', 'revenue': 2780.50},
        #     'total_months_analyzed': 12
        # }

        # format forecasts for API Response
        # convert dataframe to list of dicts
        forecasts = [
            {
                "month": row['month'],
                "predicted_revenue": round(row['predicted_revenue'], 2),
                "lower_bound": round(row['lower_bound'], 2),
                "upper_bound": round(row['upper_bound'], 2),
                "confidence": int(row['confidence'])
            }
            for _, row in forecast_df.iterrows()
        ]

        return {
            "forecasts": forecasts,
            "insights": insights
        }


ml_models = MLModels()
