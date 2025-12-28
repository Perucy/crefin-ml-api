"""
    Client Segmentation using K-Means Clustering
    Automatically groups similar clients into segments
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import joblib

class ClientSegmenter:
    """
        Segments clients into groups based on payment behavior
        Uses K-Means clustering for automatic grouping
    """

    def __init__(self, n_segments: int = 4):
        """
            Initialize segmenter
            Args:
                n_segments: Number of client segments to create (default: 4)
        """
        self.n_segments = n_segments
        self.kmeans = None
        self.scaler = StandardScaler()
        self.segment_profiles = {}

    def fit(self, clients_df: pd.DataFrame) -> 'ClientSegmenter':
        """
            Train the segmentation model on client data

            Args:
                clients_df: DataFrame with client features

            Returns:
                self (for method chaining)
        """
        print(f"\n🎯 Training segmentation model with {self.n_segments} segments...")

        # select features for clustering
        feature_columns = [
            'client_avg_payment_days',
            'client_payment_std',
            'client_late_payment_rate',
            'risk_score'
        ]

        # extract features
        X = clients_df[feature_columns].values

        # standardize features 
        X_scaled = self.scaler.fit_transform(X)

        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_segments,
            random_state=42,
            n_init=10,
            max_iter=300
        )

        cluster_labels = self.kmeans.fit_predict(X_scaled)

        # add cluster labels to determine
        clients_df['segment'] = cluster_labels

        # profile each segment
        self._profile_segments(clients_df)

        print(f"✅ Segmentation complete!")
        print(f"   Segments created: {self.n_segments}")
        
        return self
    
    def predict(self, client_features: Dict) -> Tuple[int, str, Dict]:
        """
            Predict which segment a client belongs to

            Args:
                client_features: Dict with client features
            Returns:
                Tuple of (segment_id, segment_name, segment_profile)
        """
        # extract features in correct order
        features = np.array([[
            client_features['client_avg_payment_days'],
            client_features['client_payment_std'],
            client_features['client_late_payment_rate'],
            client_features['risk_score']
        ]])

        # scale features
        features_scaled = self.scaler.transform(features)

        # predict segment
        segment_id = int(self.kmeans.predict(features_scaled)[0])

        # get segment profile
        profile = self.segment_profiles[segment_id]

        return segment_id, profile['name'], profile
    
    def _profile_segments(self, clients_df: pd.DataFrame):
        """
            Analyze and name each segment based on characteristics
        """
        print(f"\n📊 Profiling segments...")
        
        for segment_id in range(self.n_segments):
            segment_clients = clients_df[clients_df['segment'] == segment_id]
            
            if len(segment_clients) == 0:
                continue
            
            # Calculate statistics
            avg_payment = segment_clients['client_avg_payment_days'].mean()
            avg_std = segment_clients['client_payment_std'].mean()
            avg_risk = segment_clients['risk_score'].mean()
            late_rate = segment_clients['client_late_payment_rate'].mean()
            count = len(segment_clients)
            
            # Determine segment name based on characteristics
            name = self._name_segment(avg_payment, avg_std, avg_risk)
            
            # Create profile
            profile = {
                'segment_id': segment_id,
                'name': name,
                'count': count,
                'avg_payment_days': round(avg_payment, 1),
                'avg_consistency': round(avg_std, 1),
                'avg_risk_score': round(avg_risk, 1),
                'late_payment_rate': round(late_rate * 100, 1),
                'description': self._describe_segment(name, avg_payment, avg_std, avg_risk),
                'recommendation': self._recommend_for_segment(name, avg_risk)
            }
            
            self.segment_profiles[segment_id] = profile
            
            print(f"\n   Segment {segment_id}: {name}")
            print(f"      Clients: {count}")
            print(f"      Avg Payment: {profile['avg_payment_days']} days")
            print(f"      Risk Score: {profile['avg_risk_score']}/100")

    def _name_segment(self, avg_payment: float, avg_std: float, avg_risk: float) -> str:
        """
        Give each segment a meaningful name
        """
        
        # Fast payers
        if avg_payment < 20:
            if avg_std < 5:
                return "Fast & Reliable"
            else:
                return "Fast But Inconsistent"
        
        # Medium payers
        elif avg_payment < 35:
            if avg_std < 6:
                return "Steady Medium"
            else:
                return "Medium & Variable"
        
        # Slow payers
        else:
            if avg_std < 8:
                return "Slow But Predictable"
            else:
                return "Slow & Chaotic"
    
    def _describe_segment(self, name: str, avg_payment: float, avg_std: float, avg_risk: float) -> str:
        """
        Create description for segment
        """
        
        descriptions = {
            "Fast & Reliable": f"Your best clients! Pay quickly (avg {avg_payment:.0f} days) and consistently. Low risk.",
            "Fast But Inconsistent": f"Pay fast on average ({avg_payment:.0f} days) but timing varies. Monitor closely.",
            "Steady Medium": f"Reliable clients who pay in {avg_payment:.0f} days consistently. Standard terms work well.",
            "Medium & Variable": f"Average speed ({avg_payment:.0f} days) but inconsistent. Watch for delays.",
            "Slow But Predictable": f"Take {avg_payment:.0f} days but consistent. Plan accordingly.",
            "Slow & Chaotic": f"Slow ({avg_payment:.0f} days) and unpredictable. High risk, require deposits."
        }
        
        return descriptions.get(name, f"Clients paying in ~{avg_payment:.0f} days")
    
    def _recommend_for_segment(self, name: str, avg_risk: float) -> str:
        """
        Provide recommendations for each segment
        """
        
        recommendations = {
            "Fast & Reliable": "Offer preferential rates or terms. Build long-term relationships.",
            "Fast But Inconsistent": "Standard terms. Send early reminders before due dates.",
            "Steady Medium": "Standard payment terms (Net-30). Reliable business partners.",
            "Medium & Variable": "Consider Net-15 terms. Monitor payment patterns closely.",
            "Slow But Predictable": "Use Net-45 or Net-60 terms. Plan cash flow around longer cycles.",
            "Slow & Chaotic": "Require 50% deposits or milestone payments. High risk of late payment."
        }
        
        return recommendations.get(name, "Monitor payment behavior and adjust terms as needed.")
    
    def get_segment_summary(self) -> pd.DataFrame:
        """
        Get summary of all segments
        
        Returns:
            DataFrame with segment profiles
        """
        
        summary = []
        for segment_id, profile in self.segment_profiles.items():
            summary.append({
                'Segment ID': segment_id,
                'Name': profile['name'],
                'Clients': profile['count'],
                'Avg Payment (days)': profile['avg_payment_days'],
                'Consistency (std)': profile['avg_consistency'],
                'Risk Score': profile['avg_risk_score'],
                'Late Rate (%)': profile['late_payment_rate']
            })
        
        return pd.DataFrame(summary)


# Export
__all__ = ['ClientSegmenter']

