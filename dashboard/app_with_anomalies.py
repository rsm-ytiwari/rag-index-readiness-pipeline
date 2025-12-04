"""
RAG Index-Readiness Dashboard - With Anomaly Detection
Part 2: Production Features (Hours 1-7)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anomaly_detector import detect_quality_anomalies

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RAG Index-Readiness Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_pdf_report(df):
    """Generate PDF quality report."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    title = Paragraph("<b>RAG Index-Readiness Quality Report</b>", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 0.3*inch))
    
    summary_text = f"""
    <b>Executive Summary</b><br/>
    Total Reviews Analyzed: {len(df):,}<br/>
    Average Quality Score: {df['index_readiness_score'].mean():.1f}/100<br/>
    Index-Ready Reviews: {(df['index_ready']).sum():,} ({(df['index_ready']).sum()/len(df)*100:.1f}%)<br/>
    Reviews Needing Manual Review: {(df['recommendation'] == 'review').sum():,} ({(df['recommendation'] == 'review').sum()/len(df)*100:.1f}%)<br/>
    Duplicate Rate: {(df['is_duplicate']).sum()/len(df)*100:.1f}%<br/>
    PII Risk: {(df['has_pii']).sum()/len(df)*100:.1f}%<br/>
    """
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    elements.append(Paragraph("<b>Quality Distribution</b>", styles['Heading2']))
    
    quality_data = [
        ['Metric', 'Count', 'Percentage'],
        ['Index Ready', f"{(df['recommendation'] == 'index').sum():,}", f"{(df['recommendation'] == 'index').sum()/len(df)*100:.1f}%"],
        ['Needs Review', f"{(df['recommendation'] == 'review').sum():,}", f"{(df['recommendation'] == 'review').sum()/len(df)*100:.1f}%"],
        ['Reject', f"{(df['recommendation'] == 'reject').sum():,}", f"{(df['recommendation'] == 'reject').sum()/len(df)*100:.1f}%"],
    ]
    
    table = Table(quality_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 0.3*inch))
    
    elements.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
    
    recommendations = f"""
    1. <b>Index {(df['index_ready']).sum():,} reviews immediately</b> - These meet quality standards<br/>
    2. <b>Review {(df['has_pii']).sum():,} reviews with PII</b> - Consider redaction before indexing<br/>
    3. <b>Monitor chunk quality</b> - {(df['chunk_quality_flag'] == 'too_short').sum():,} reviews are too short<br/>
    4. <b>Estimated Cost Savings</b> - Filtering saves ~${(len(df) - (df['index_ready']).sum()) * 0.25 / 1000 * 12:,.0f}/year in vector DB costs<br/>
    """
    elements.append(Paragraph(recommendations, styles['Normal']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ============================================================================
# DATA LOADING WITH CACHING
# ============================================================================

@st.cache_data
def load_data():
    """Load Gold Parquet with caching for performance."""
    data_path = Path("../data/gold/reviews_featured.parquet")
    
    if not data_path.exists():
        data_path = Path("data/gold/reviews_featured.parquet")
    
    df = pd.read_parquet(data_path)
    
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    
    df['pii_types_list'] = df['pii_types'].apply(
        lambda x: json.loads(x) if x and x != '[]' else []
    )
    
    return df

# Load data
try:
    df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ============================================================================
# ANOMALY DETECTION
# ============================================================================

@st.cache_data
def detect_anomalies(df):
    """Detect quality anomalies with caching."""
    anomaly_df, summary = detect_quality_anomalies(df)
    return anomaly_df, summary

anomaly_df, anomaly_summary = detect_anomalies(df)

# ============================================================================
# ANOMALY ALERT BANNER
# ============================================================================

if anomaly_summary['anomaly_count'] > 0:
    st.warning(f"⚠️ **Quality Alert:** {anomaly_summary['anomaly_count']} anomalous period(s) detected ({anomaly_summary['anomaly_rate']:.1f}% of time periods)")
    
    with st.expander("🔍 View Anomaly Details"):
        if anomaly_summary['low_quality_periods']:
            st.markdown("**Low Quality Periods:**")
            for period in anomaly_summary['low_quality_periods']:
                st.markdown(f"- {period}")
        if anomaly_summary['high_quality_periods']:
            st.markdown("**High Quality Periods:**")
            for period in anomaly_summary['high_quality_periods']:
                st.markdown(f"- {period}")

# ============================================================================
# SIDEBAR - FILTERS
# ============================================================================

st.sidebar.header("🔍 Filters")

st.sidebar.subheader("Date Range")
min_date = df['date'].min().date()
max_date = df['date'].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

st.sidebar.subheader("Star Rating")
star_options = sorted(df['stars'].unique())
selected_stars = st.sidebar.multiselect(
    "Select star ratings",
    options=star_options,
    default=star_options
)

st.sidebar.subheader("Quality Tier")
quality_options = ['All', 'Index Ready', 'Needs Review', 'Reject']
selected_quality = st.sidebar.selectbox(
    "Select quality tier",
    options=quality_options,
    index=0
)

# Apply filters
filtered_df = df.copy()

filtered_df = filtered_df[
    (filtered_df['date'].dt.date >= start_date) & 
    (filtered_df['date'].dt.date <= end_date)
]

if selected_stars:
    filtered_df = filtered_df[filtered_df['stars'].isin(selected_stars)]

if selected_quality != 'All':
    if selected_quality == 'Index Ready':
        filtered_df = filtered_df[filtered_df['recommendation'] == 'index']
    elif selected_quality == 'Needs Review':
        filtered_df = filtered_df[filtered_df['recommendation'] == 'review']
    elif selected_quality == 'Reject':
        filtered_df = filtered_df[filtered_df['recommendation'] == 'reject']

st.sidebar.markdown("---")
st.sidebar.info(f"**Filtered:** {len(filtered_df):,} / {len(df):,} reviews ({len(filtered_df)/len(df)*100:.1f}%)")

# ============================================================================
# COST CALCULATOR WIDGET
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.header("💰 Cost Impact Calculator")

vector_db_cost_per_1k = st.sidebar.slider(
    "Vector DB Cost ($/1K embeddings)",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05,
    help="Typical costs: Pinecone $0.25/1K, Weaviate $0.20/1K"
)

reviews_to_index = st.sidebar.number_input(
    "Reviews to Index (millions)",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
    help="Total number of reviews you plan to index"
)

total_reviews = reviews_to_index * 1_000_000
index_ready_rate = (df['index_ready']).sum() / len(df)

reviews_filtered = total_reviews * (1 - index_ready_rate)
cost_without_filter = total_reviews * vector_db_cost_per_1k / 1000
cost_with_filter = total_reviews * index_ready_rate * vector_db_cost_per_1k / 1000
monthly_savings = cost_without_filter - cost_with_filter
annual_savings = monthly_savings * 12

st.sidebar.metric(
    "💵 Monthly Savings",
    f"${monthly_savings:,.0f}",
    help=f"Saves {reviews_filtered:,.0f} embeddings/month"
)

st.sidebar.metric(
    "📈 Annual ROI",
    f"${annual_savings:,.0f}",
    delta=f"{reviews_filtered/1_000_000:.1f}M embeddings saved"
)

st.sidebar.success(f"✅ Filter removes {(1-index_ready_rate)*100:.1f}% of low-quality reviews")

# ============================================================================
# BEFORE/AFTER COMPARISON TOGGLE
# ============================================================================

st.sidebar.markdown("---")
comparison_mode = st.sidebar.checkbox(
    "📊 Show Before/After Comparison",
    value=False,
    help="Compare metrics with and without quality filtering"
)

# ============================================================================
# MAIN DASHBOARD - HEADER
# ============================================================================

st.title("📊 RAG Index-Readiness Dashboard")
st.markdown("### Yelp Reviews Data Quality Analysis")

# Export buttons
col1, col2, col3 = st.columns(3)

with col1:
    ready_reviews = filtered_df[filtered_df['index_ready']].copy()
    ready_csv = ready_reviews[['review_id', 'text', 'stars', 'date', 'index_readiness_score']].to_csv(index=False)
    
    st.download_button(
        label="📥 Download Index-Ready Reviews",
        data=ready_csv,
        file_name="index_ready_reviews.csv",
        mime="text/csv",
        help=f"{len(ready_reviews):,} reviews ready for vector DB"
    )

with col2:
    pii_reviews = filtered_df[filtered_df['has_pii']].copy()
    pii_csv = pii_reviews[['review_id', 'text', 'pii_types', 'index_readiness_score']].to_csv(index=False)
    
    st.download_button(
        label="📥 Download PII Reviews",
        data=pii_csv,
        file_name="pii_reviews_for_review.csv",
        mime="text/csv",
        help=f"{len(pii_reviews):,} reviews with potential PII"
    )

with col3:
    pdf_buffer = generate_pdf_report(filtered_df)
    
    st.download_button(
        label="📥 Download Quality Report (PDF)",
        data=pdf_buffer,
        file_name="quality_report.pdf",
        mime="application/pdf",
        help="Executive summary PDF report"
    )

st.markdown("---")

# (Continue with rest of dashboard code from previous version...)
# For brevity, showing just the key parts. The rest remains the same.

st.caption("RAG Index-Readiness Pipeline | MGTA 452 Project | With Anomaly Detection")
