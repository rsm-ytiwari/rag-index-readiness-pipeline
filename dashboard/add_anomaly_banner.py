# Add this code to the top of dashboard/app.py after imports

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from anomaly_detector import detect_quality_anomalies

# After loading data, add:

@st.cache_data
def detect_anomalies(df):
    """Detect quality anomalies with caching."""
    anomaly_df, summary = detect_quality_anomalies(df)
    return anomaly_df, summary

anomaly_df, anomaly_summary = detect_anomalies(df)

# Add anomaly banner after title, before export buttons:

if anomaly_summary['anomaly_count'] > 0:
    st.warning(f"⚠️ **Quality Alert:** {anomaly_summary['anomaly_count']} anomalous period(s) detected ({anomaly_summary['anomaly_rate']:.1f}% of time periods)")
    
    with st.expander("🔍 View Anomaly Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            if anomaly_summary['low_quality_periods']:
                st.markdown("**⚠️ Low Quality Periods:**")
                for period in anomaly_summary['low_quality_periods']:
                    st.markdown(f"- {period}")
        
        with col2:
            if anomaly_summary['high_quality_periods']:
                st.markdown("**✅ High Quality Periods:**")
                for period in anomaly_summary['high_quality_periods']:
                    st.markdown(f"- {period}")
        
        st.markdown("**📊 Detection Method:** Z-score (threshold: 2.0 std devs)")
