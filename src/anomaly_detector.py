"""
Anomaly Detection Module

Detects anomalies in data quality metrics over time using statistical methods.
Flags periods where quality drops significantly below normal levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

class QualityAnomalyDetector:
    """
    Detect anomalies in review quality metrics using statistical methods.
    
    Methods:
    - Z-score method (outliers > 2 std devs from mean)
    - IQR method (outliers outside 1.5 * IQR)
    - Moving average deviation
    """
    
    def __init__(self, method='zscore', threshold=2.0):
        """
        Initialize anomaly detector.
        
        Args:
            method: Detection method ('zscore', 'iqr', or 'moving_avg')
            threshold: Threshold for anomaly detection (default: 2.0 for z-score)
        """
        self.method = method
        self.threshold = threshold
    
    def detect_anomalies(self, df: pd.DataFrame, metric_col: str, time_col: str = 'year_month') -> pd.DataFrame:
        """
        Detect anomalies in a time series metric.
        
        Args:
            df: DataFrame with time series data
            metric_col: Column name of metric to check
            time_col: Column name for time periods
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        # Aggregate by time period
        agg_df = df.groupby(time_col).agg({
            metric_col: ['mean', 'std', 'count']
        }).reset_index()
        
        agg_df.columns = [time_col, 'metric_mean', 'metric_std', 'review_count']
        
        # Apply detection method
        if self.method == 'zscore':
            agg_df = self._zscore_detection(agg_df, 'metric_mean')
        elif self.method == 'iqr':
            agg_df = self._iqr_detection(agg_df, 'metric_mean')
        elif self.method == 'moving_avg':
            agg_df = self._moving_avg_detection(agg_df, 'metric_mean')
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return agg_df
    
    def _zscore_detection(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Z-score method: Flag if |z| > threshold."""
        mean = df[col].mean()
        std = df[col].std()
        
        df['zscore'] = (df[col] - mean) / std if std > 0 else 0
        df['is_anomaly'] = np.abs(df['zscore']) > self.threshold
        df['anomaly_severity'] = np.abs(df['zscore'])
        df['anomaly_type'] = df['zscore'].apply(
            lambda z: 'low' if z < -self.threshold else ('high' if z > self.threshold else 'normal')
        )
        
        return df
    
    def _iqr_detection(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """IQR method: Flag if outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df['is_anomaly'] = (df[col] < lower_bound) | (df[col] > upper_bound)
        df['anomaly_severity'] = df.apply(
            lambda row: max(0, (lower_bound - row[col]) / IQR) if row[col] < lower_bound 
                       else max(0, (row[col] - upper_bound) / IQR),
            axis=1
        )
        df['anomaly_type'] = df.apply(
            lambda row: 'low' if row[col] < lower_bound 
                       else ('high' if row[col] > upper_bound else 'normal'),
            axis=1
        )
        
        return df
    
    def _moving_avg_detection(self, df: pd.DataFrame, col: str, window=3) -> pd.DataFrame:
        """Moving average method: Flag if deviation from MA exceeds threshold."""
        df = df.sort_values('year_month').reset_index(drop=True)
        df['moving_avg'] = df[col].rolling(window=window, center=False).mean()
        df['deviation'] = (df[col] - df['moving_avg']) / df['moving_avg']
        
        df['is_anomaly'] = np.abs(df['deviation']) > (self.threshold / 10)  # Scale threshold
        df['anomaly_severity'] = np.abs(df['deviation'])
        df['anomaly_type'] = df['deviation'].apply(
            lambda d: 'low' if d < -(self.threshold/10) 
                     else ('high' if d > (self.threshold/10) else 'normal')
        )
        
        return df
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of detected anomalies.
        
        Returns:
            Dictionary with anomaly statistics
        """
        if 'is_anomaly' not in df.columns:
            return {
                'total_periods': len(df),
                'anomaly_count': 0,
                'anomaly_rate': 0.0,
                'anomalous_periods': []
            }
        
        anomalies = df[df['is_anomaly']]
        
        return {
            'total_periods': len(df),
            'anomaly_count': len(anomalies),
            'anomaly_rate': len(anomalies) / len(df) * 100 if len(df) > 0 else 0,
            'anomalous_periods': anomalies['year_month'].tolist() if 'year_month' in anomalies.columns else [],
            'low_quality_periods': anomalies[anomalies['anomaly_type'] == 'low']['year_month'].tolist() if 'year_month' in anomalies.columns else [],
            'high_quality_periods': anomalies[anomalies['anomaly_type'] == 'high']['year_month'].tolist() if 'year_month' in anomalies.columns else []
        }

def detect_quality_anomalies(df: pd.DataFrame, metric='index_readiness_score') -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to detect quality anomalies.
    
    Args:
        df: DataFrame with review data (must have 'date' column)
        metric: Metric column to analyze (default: 'index_readiness_score')
    
    Returns:
        Tuple of (anomaly_df, summary_dict)
    """
    # Add year_month if not present
    if 'year_month' not in df.columns:
        df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')
    
    # Initialize detector
    detector = QualityAnomalyDetector(method='zscore', threshold=2.0)
    
    # Detect anomalies
    anomaly_df = detector.detect_anomalies(df, metric)
    summary = detector.get_anomaly_summary(anomaly_df)
    
    return anomaly_df, summary

if __name__ == '__main__':
    # Test anomaly detection
    import sys
    from pathlib import Path
    
    # Load gold data
    data_path = Path(__file__).parent.parent / 'data' / 'gold' / 'reviews_featured.parquet'
    df = pd.read_parquet(data_path)
    
    print("="*70)
    print("ANOMALY DETECTION TEST")
    print("="*70)
    
    # Detect anomalies
    anomaly_df, summary = detect_quality_anomalies(df)
    
    print(f"\n📊 Anomaly Summary:")
    print(f"  Total periods: {summary['total_periods']}")
    print(f"  Anomalies detected: {summary['anomaly_count']} ({summary['anomaly_rate']:.1f}%)")
    
    if summary['low_quality_periods']:
        print(f"\n⚠️  Low quality periods:")
        for period in summary['low_quality_periods']:
            print(f"    - {period}")
    
    if summary['high_quality_periods']:
        print(f"\n✅ High quality periods:")
        for period in summary['high_quality_periods']:
            print(f"    - {period}")
    
    print("\n📈 Anomaly Details:")
    if len(anomaly_df[anomaly_df['is_anomaly']]) > 0:
        print(anomaly_df[anomaly_df['is_anomaly']][['year_month', 'metric_mean', 'zscore', 'anomaly_type', 'review_count']])
    else:
        print("  No anomalies detected (data quality is consistent)")
    
    print("\n" + "="*70)
