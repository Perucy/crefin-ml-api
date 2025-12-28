"""
    Revenue Forecasting using Time Series Analysis
    Predicts future monthly revenue based on hsitorical patterns
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

class RevenueForecaster:
    """
    Forecasts future revenue based on historical invoice data
    Uses trend analysis and seasonal patterns
    """

    def __init__(self):
        self.model = None
        self.monthly_stats = None
        self.trend_model = LinearRegression()
        self.is_fitted = False

    def fit(self, invoices_df: pd.DataFrame) -> 'RevenueForecaster':
        """
        Train forecasting model on historical invoice data
        
        Args:
            invoices_df: DataFrame with columns [issue_data, amount, payment_days]

        Returns:
            self(for method chaining)
        """
        print("\n📊 Analyzing historical revenue patterns...")

        # ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(invoices_df['issue_date']):
            invoices_df['issue_date'] = pd.to_datetime(invoices_df['issue_date'])

        # calculate actual payment date
        invoices_df['payment_date'] = invoices_df['issue_date'] + pd.to_timedelta(
            invoices_df['payment_days'], unit='D'
        )

        # group by month (using payment date for revenue recognition)
        invoices_df['year_month'] = invoices_df['payment_date'].dt.to_period('M')

        # Calculate monthly revenue
        monthly_revenue = invoices_df.groupby('year_month').agg({
            'amount': ['sum', 'count', 'mean', 'std']
        }).reset_index()

        monthly_revenue.columns = ['year_month', 'total_revenue', 'invoice_count', 'avg_income', 'std_invoice']

        # fill missing std with 0 for months with single invoice
        monthly_revenue['std_invoice'] = monthly_revenue['std_invoice'].fillna(0)

        # convert period to datetime for modeling
        monthly_revenue['date'] = monthly_revenue['year_month'].dt.to_timestamp()

        # time-based features
        monthly_revenue['month_num'] = monthly_revenue['date'].dt.month
        monthly_revenue['month_index'] = range(len(monthly_revenue))

        # fit trend model
        X = monthly_revenue[['month_index']].values
        y = monthly_revenue['total_revenue'].values

        self.trend_model.fit(X, y)

        # store monthly stats
        self.monthly_stats = monthly_revenue
        self.is_fitted = True

        # calculate growth rate
        if len(monthly_revenue) >= 2:
            recent_avg = monthly_revenue.tail(3)['total_revenue'].mean()
            older_avg = monthly_revenue.head(3)['total_revenue'].mean()
            self.growth_rate = ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0
        else:
            self.growth_rate = 0

        print(f"✅ Analyzed {len(monthly_revenue)} months of revenue data")
        print(f"   Growth rate: {self.growth_rate:+.1f}%")

        return self
    
    def forecast(self, months_ahead: int = 3) -> pd.DataFrame:
        """
        Forecast revenue for future months
        
        Args:
            months_ahead: Number of months to forecast
            
        Returns:
            DataFrame with forecasted revenue
        """

        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        print(f"\n🔮 Forecasting revenue for next {months_ahead} months...")

        # get the last known month
        last_month_index = self.monthly_stats['month_index'].max()
        last_date = self.monthly_stats['date'].max()

        # generate future month indices
        future_indices = np.arange(
            last_month_index + 1,
            last_month_index + 1 + months_ahead
        ).reshape(-1, 1)

        # predict using trend
        trend_predictions = self.trend_model.predict(future_indices)

        # calculate confidence intervals based on historical variance
        historical_std = self.monthly_stats['total_revenue'].std()

        # create forecast dataframe
        forecasts = []
        for i, pred in enumerate(trend_predictions):
            future_date = last_date + pd.DateOffset(months=i+1)

            # adjust for seasonality (simple: look at same month in past)
            month_num = future_date.month
            same_month_history = self.monthly_stats[
                self.monthly_stats['month_num'] == month_num
            ]

            if len(same_month_history) > 0:
                seasonal_factor = same_month_history['total_revenue'].mean() / self.monthly_stats['total_revenue'].mean()
                seasonal_adjusted = pred * seasonal_factor
            else:
                seasonal_adjusted = pred
            
            # calculate confidence interval (+/- 1.5  std for ~86% confidence)
            confidence_range = historical_std * 1.5 * (1 + i * 0.1) # wider for further future

            forecasts.append({
                'month': future_date.strftime('%Y-%m'),
                'date': future_date,
                'predicted_revenue': max(0, seasonal_adjusted),
                'lower_bound': max(0, seasonal_adjusted - confidence_range),
                'upper_bound': seasonal_adjusted + confidence_range,
                'confidence': max(50, 90 - (i * 5))
            })
    
        forecast_df = pd.DataFrame(forecasts)

        print(f"✅ Forecast generated")

        return forecast_df
    
    def get_insights(self) -> Dict:
        """
        Generate business insights from revenue data
        
        Returns:
            Dictionary with insights
            
        """

        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating insights")
        
        stats = self.monthly_stats

        # calculate key metrics
        current_month_revenue = stats.iloc[-1]['total_revenue']
        avg_monthly_revenue = stats['total_revenue'].mean()
        best_month = stats.loc[stats['total_revenue'].idxmax()]
        worst_month = stats.loc[stats['total_revenue'].idxmin()]

        # volatility
        volatility = stats['total_revenue'].std() / stats['total_revenue'].mean() * 100

        # trend direction
        if self.growth_rate > 10:
            trend = "Strong Growth"
        elif self.growth_rate > 0:
            trend = "Moderate Growth"
        elif self.growth_rate > -10:
            trend = "Slight Decline"
        else:
            trend = "Declining"

        # revenue consistency
        if volatility < 20:
            consistency = "Very Stable"
        elif volatility < 40:
            consistency = "Moderately Stable"
        else:
            consistency = "Volatile"

        return {
            'current_month_revenue': current_month_revenue,
            'avg_monthly_revenue': avg_monthly_revenue,
            'growth_rate': self.growth_rate,
            'trend': trend,
            'volatility': volatility,
            'consistency': consistency,
            'best_month': {
                'date': best_month['date'].strftime('%Y-%m'),
                'revenue': best_month['total_revenue']
            },
            'worst_month': {
                'date': worst_month['date'].strftime('%Y-%m'),
                'revenue': worst_month['total_revenue']
            },
            'total_months_analyzed': len(stats)
        }
    
    def get_monthly_breakdown(self) -> pd.DataFrame:
        """
        Get detailed monthly breakdown
        
        Returns:
            DataFrame with monthly statistics
        """

        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting monthly breakdown")
        
        breakdown = self.monthly_stats[[
            'year_month', 'total_revenue', 'invoice_count', 
            'avg_income', 'std_invoice'
        ]].copy()

        return breakdown
    
# export
__all__ = ['RevenueForecaster']
