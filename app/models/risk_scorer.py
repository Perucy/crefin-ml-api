"""
    Client risk scoring model
    Calculates risk scores based on payment behavior
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class ClientRiskScorer:
    """
        Calculates client risk scores based on payment history
        Risk score:
            O - 100 (HIGHER = RISKIER)
            0 - 30: Low Risk
            31 - 60: Medium
            61 - 100: High
    """
    def __init__(self):
        # weight for different risk factors
        self.weights = {
            'late_payment': 0.40,
            'payment_speed': 0.25,
            'consistency': 0.20,
            'trend': 0.10,
            'experience': 0.05
        }
    
    def calculate_risk_score(self, client_features: Dict) -> Dict:
        """
            Calculate risk score for a client
            Args:
                client_features: Dict with client payment features
            Returns:
                Dict with risk score and breakdown
        """
        # calc individual risk components
        late_risk = self._calculate_late_payment_risk(client_features)
        speed_risk = self._calculate_speed_risk(client_features)
        consistency_risk = self._calculate_consistency_risk(client_features)
        trend_risk = self._calculate_trend_risk(client_features)
        experience_risk = self._calculate_experience_risk(client_features)

        # weighted sum
        total_risk = (
            late_risk * self.weights['late_payment'] +
            speed_risk * self.weights['payment_speed'] +
            consistency_risk * self.weights['consistency'] +
            trend_risk * self.weights['trend'] +
            experience_risk * self.weights['experience']
        )

        # Convert to 0-100 scale
        risk_score = int(total_risk * 100)
        
        # Determine risk level
        risk_level = self._get_risk_level(risk_score)
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_breakdown': {
                'late_payment_risk': round(late_risk * 100, 1),
                'speed_risk': round(speed_risk * 100, 1),
                'consistency_risk': round(consistency_risk * 100, 1),
                'trend_risk': round(trend_risk * 100, 1),
                'experience_risk': round(experience_risk * 100, 1)
            },
            'recommendation': self._get_recommendation(risk_score)
        }
        

    def _calculate_late_payment_risk(self, features: Dict) -> float:
        """
            Calculate risk based on late payment rate
            
            Late payment rate (% of invoices paid after 30 days):
            - 0-10%: Low risk (0.0-0.2)
            - 10-30%: Medium risk (0.2-0.5)
            - 30-60%: High risk (0.5-0.8)
            - 60%+: Very high risk (0.8-1.0)
        """
        late_rate = features.get('client_late_payment_rate', 0.5)
        
        if late_rate <= 0.1:
            return 0.1
        elif late_rate <= 0.3:
            return 0.2 + (late_rate - 0.1) * 1.5  # Scale 0.1-0.3 → 0.2-0.5
        elif late_rate <= 0.6:
            return 0.5 + (late_rate - 0.3) * 1.0  # Scale 0.3-0.6 → 0.5-0.8
        else:
            return 0.8 + min((late_rate - 0.6) * 0.5, 0.2)  # 0.6+ → 0.8-1.0

    def _calculate_speed_risk(self, features: Dict) -> float:
        """
            Calculate risk based on average payment speed
            
            Average payment days:
            - 0-20 days: Low risk (0.0-0.2)
            - 20-35 days: Medium risk (0.2-0.5)
            - 35-50 days: High risk (0.5-0.8)
            - 50+ days: Very high risk (0.8-1.0)
        """
        avg_days = features.get('client_avg_payment_days', 30)
        
        if avg_days <= 20:
            return avg_days / 100  # 0-20 days → 0.0-0.2
        elif avg_days <= 35:
            return 0.2 + (avg_days - 20) * 0.02  # 20-35 → 0.2-0.5
        elif avg_days <= 50:
            return 0.5 + (avg_days - 35) * 0.02  # 35-50 → 0.5-0.8
        else:
            return 0.8 + min((avg_days - 50) * 0.004, 0.2)  # 50+ → 0.8-1.0

    def _calculate_consistency_risk(self, features: Dict) -> float:
        """
            Calculate risk based on payment consistency (std deviation)
            
            Payment std (variance in days):
            - 0-5 days: Very consistent, low risk (0.0-0.2)
            - 5-10 days: Somewhat consistent, medium risk (0.2-0.5)
            - 10-20 days: Inconsistent, high risk (0.5-0.8)
            - 20+ days: Very inconsistent, very high risk (0.8-1.0)
        """
        std = features.get('client_payment_std', 10)
        
        if std <= 5:
            return std / 25  # 0-5 → 0.0-0.2
        elif std <= 10:
            return 0.2 + (std - 5) * 0.06  # 5-10 → 0.2-0.5
        elif std <= 20:
            return 0.5 + (std - 10) * 0.03  # 10-20 → 0.5-0.8
        else:
            return 0.8 + min((std - 20) * 0.01, 0.2)  # 20+ → 0.8-1.0

    def _calculate_trend_risk(self, features: Dict) -> float:
        """
            Calculate risk based on payment trend
            
            Trend (positive = getting worse):
            - Negative (improving): Low risk (0.0-0.3)
            - Zero (stable): Medium risk (0.4-0.6)
            - Positive (worsening): High risk (0.7-1.0)
        """
        trend = features.get('client_payment_trend', 0)

        if trend < -5:
            return 0.0  # Improving fast
        elif trend < 0:
            return 0.3 - (trend / 15)  # Improving slowly
        elif trend == 0:
            return 0.5  # Stable
        elif trend <= 10:
            return 0.5 + (trend * 0.03)  # Getting worse
        else:
            return 0.8 + min((trend - 10) * 0.02, 0.2)  

    def _calculate_experience_risk(self, features: Dict) -> float:
        """
            Calculate risk based on relationship history
            
            Total invoices (more history = less risk):
            - 0-2: New client, high risk (0.8-1.0)
            - 3-5: Some history, medium risk (0.5-0.8)
            - 6-10: Good history, low-medium risk (0.3-0.5)
            - 10+: Established, low risk (0.0-0.3)
        """
        total = features.get('client_total_invoices', 0)
        
        if total == 0:
            return 1.0  # New client
        elif total <= 2:
            return 0.9
        elif total <= 5:
            return 0.7 - (total - 2) * 0.06  # 3-5 → 0.7-0.5
        elif total <= 10:
            return 0.5 - (total - 5) * 0.04  # 6-10 → 0.5-0.3
        else:
            return max(0.3 - (total - 10) * 0.02, 0.0)  # 10+ → 0.3-0.0

    def _get_risk_level(self, score: int) -> str:
        """Convert numeric score to risk level"""
        if score <= 30:
            return 'low'
        elif score <= 60:
            return 'medium'
        else:
            return 'high'
    
    def _get_recommendation(self, score: int) -> str:
        """Get action recommendation based on risk score"""
        if score <= 30:
            return "Safe to work with. Reliable payment history."
        elif score <= 45:
            return "Generally safe. Monitor payment behavior."
        elif score <= 60:
            return "Moderate risk. Consider payment terms or deposits."
        elif score <= 75:
            return "High risk. Require deposits or milestone payments."
        else:
            return "Very high risk. Avoid or require full payment upfront."
    
    def calculate_risk_scores_batch(self, clients_df: pd.DataFrame) -> pd.DataFrame:
        """
            Calculate risk scores for multiple clients
            
            Args:
                clients_df: DataFrame with client features
                
            Returns:
                DataFrame with added risk scores
        """
        results = []
        
        for idx, row in clients_df.iterrows():
            risk_data = self.calculate_risk_score(row.to_dict())
            results.append({
                'client_id': row.get('client_id'),
                'risk_score': risk_data['risk_score'],
                'risk_level': risk_data['risk_level'],
                'recommendation': risk_data['recommendation']
            })
        
        return pd.DataFrame(results)


# Export
__all__ = ['ClientRiskScorer']