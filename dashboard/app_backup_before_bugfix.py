"""
RAG Index-Readiness Dashboard - Complete Version
With Anomaly Detection, Cost Calculator, Search, and All Features
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
    4. <b>Estimated Cost Savings</b> - Filtering saves ~${(len(df) - (df['index_ready']).sum()) * 0.25 / 1000 * 12:,.0f}/year<br/>
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
    data_path = Path("data/gold/reviews_featured.parquet")
    
    if not data_path.exists():
        st.error(f"❌ Data file not found: {data_path}")
        st.stop()
    
    df = pd.read_parquet(data_path)
    
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    
    df['pii_types_list'] = df['pii_types'].apply(
        lambda x: json.loads(x) if x and x != '[]' else []
    )
    
    return df

# Load data
df = load_data()

# ============================================================================
# ANOMALY DETECTION
# ============================================================================

@st.cache_data
def detect_anomalies(_df):
    """Detect quality anomalies with caching."""
    anomaly_df, summary = detect_quality_anomalies(_df)
    return anomaly_df, summary

anomaly_df, anomaly_summary = detect_anomalies(df)

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
st.sidebar.info(f"**Filtered:** {len(filtered_df):,} / {len(df):,} reviews")

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
    step=1
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

# ============================================================================
# ANOMALY ALERT BANNER
# ============================================================================

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

st.markdown("---")

# ============================================================================
# EXPORT BUTTONS
# ============================================================================

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

# ============================================================================
# BEFORE/AFTER COMPARISON (if enabled)
# ============================================================================

if comparison_mode:
    st.subheader("📊 Before/After Quality Filter Comparison")
    
    col1, col2 = st.columns(2)
    
    # Calculate metrics
    all_reviews = len(df)
    filtered_reviews = (df['index_ready']).sum()
    cost_all = all_reviews * vector_db_cost_per_1k / 1000
    cost_filtered = filtered_reviews * vector_db_cost_per_1k / 1000
    
    pii_all = (df['has_pii']).sum()
    pii_filtered = (df[df['index_ready']]['has_pii']).sum()
    
    avg_score_all = df['index_readiness_score'].mean()
    avg_score_filtered = df[df['index_ready']]['index_readiness_score'].mean()
    
    with col1:
        st.markdown("#### WITHOUT Quality Filter")
        st.metric("Reviews to Index", f"{all_reviews:,}")
        st.metric("Storage Cost", f"${cost_all:,.2f}/month")
        st.metric("PII Risk", f"{pii_all:,} reviews ({pii_all/all_reviews*100:.1f}%)", delta=None, delta_color="inverse")
        st.metric("Avg Quality Score", f"{avg_score_all:.1f}/100")
        st.warning("⚠️ Indexing ALL reviews regardless of quality")
    
    with col2:
        st.markdown("#### WITH Quality Filter (score ≥ 70)")
        st.metric("Reviews to Index", f"{filtered_reviews:,}", delta=f"-{all_reviews - filtered_reviews:,} filtered")
        st.metric("Storage Cost", f"${cost_filtered:,.2f}/month", delta=f"-${cost_all - cost_filtered:,.2f}/month")
        st.metric("PII Risk", f"{pii_filtered:,} reviews ({pii_filtered/filtered_reviews*100:.1f}%)", delta=f"-{pii_all - pii_filtered:,} reviews", delta_color="inverse")
        st.metric("Avg Quality Score", f"{avg_score_filtered:.1f}/100", delta=f"+{avg_score_filtered - avg_score_all:.1f}")
        st.success("✅ Only high-quality reviews indexed")
    
    st.markdown("---")

# ============================================================================
# REVIEW SEARCH
# ============================================================================

st.subheader("🔍 Review Search")

search_query = st.text_input(
    "Search reviews by text or review_id",
    placeholder="e.g., 'pizza', 'wEXCYHTpwn1ZrZDPlLeL7A', 'terrible service'"
)

if search_query:
    search_results = filtered_df[
        filtered_df['text'].str.contains(search_query, case=False, na=False) |
        filtered_df['review_id'].str.contains(search_query, case=False, na=False)
    ].head(50)
    
    st.info(f"Found {len(search_results):,} matching reviews (showing top 50)")
    
    if len(search_results) > 0:
        display_df = search_results[['review_id', 'stars', 'index_readiness_score', 'recommendation', 'text']].copy()
        display_df['text_preview'] = display_df['text'].str[:100] + '...'
        display_df = display_df.drop('text', axis=1)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download search results
        search_csv = search_results.to_csv(index=False)
        st.download_button(
            "📥 Download Search Results",
            data=search_csv,
            file_name=f"search_results_{search_query[:20]}.csv",
            mime="text/csv"
        )

st.markdown("---")

# ============================================================================
# KPI METRICS
# ============================================================================

st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Reviews",
        f"{len(filtered_df):,}",
        delta=None
    )

with col2:
    avg_score = filtered_df['index_readiness_score'].mean()
    st.metric(
        "Avg Quality Score",
        f"{avg_score:.1f}/100",
        delta=None
    )

with col3:
    duplicate_rate = (filtered_df['is_duplicate']).sum() / len(filtered_df) * 100
    st.metric(
        "Duplicate Rate",
        f"{duplicate_rate:.1f}%",
        delta=None
    )

with col4:
    pii_rate = (filtered_df['has_pii']).sum() / len(filtered_df) * 100
    st.metric(
        "PII Risk",
        f"{pii_rate:.1f}%",
        delta=None
    )

st.markdown("---")

# ============================================================================
# QUALITY DISTRIBUTION CHART
# ============================================================================

st.subheader("📈 Quality Distribution")

quality_dist = filtered_df['recommendation'].value_counts().reset_index()
quality_dist.columns = ['recommendation', 'count']

quality_colors = {'index': '#28a745', 'review': '#ffc107', 'reject': '#dc3545'}
quality_dist['color'] = quality_dist['recommendation'].map(quality_colors)

fig = px.bar(
    quality_dist,
    x='recommendation',
    y='count',
    color='recommendation',
    color_discrete_map=quality_colors,
    labels={'count': 'Number of Reviews', 'recommendation': 'Quality Tier'},
    title=f"Review Distribution by Quality Tier (n={len(filtered_df):,})"
)

fig.update_layout(showlegend=False, height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================================
# PII ANALYSIS
# ============================================================================

st.subheader("🔒 PII Analysis")

col1, col2 = st.columns(2)

with col1:
    pii_counts = filtered_df['has_pii'].value_counts()
    fig = px.pie(
        values=pii_counts.values,
        names=['No PII' if not x else 'Has PII' for x in pii_counts.index],
        color_discrete_sequence=['#28a745', '#dc3545'],
        title="PII Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    pii_type_counts = {}
    for pii_list in filtered_df['pii_types_list']:
        for pii_type in pii_list:
            pii_type_counts[pii_type] = pii_type_counts.get(pii_type, 0) + 1
    
    if pii_type_counts:
        pii_type_df = pd.DataFrame(list(pii_type_counts.items()), columns=['PII Type', 'Count'])
        fig = px.bar(
            pii_type_df,
            x='PII Type',
            y='Count',
            title="PII Types Detected",
            color='Count',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No PII detected in filtered dataset")

st.markdown("---")

# ============================================================================
# SCORE DISTRIBUTION
# ============================================================================

st.subheader("📊 Quality Score Distribution")

fig = px.histogram(
    filtered_df,
    x='index_readiness_score',
    nbins=20,
    title="Distribution of Index-Readiness Scores",
    labels={'index_readiness_score': 'Quality Score', 'count': 'Number of Reviews'},
    color_discrete_sequence=['#1f77b4']
)

fig.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="Threshold (70)")
fig.add_vline(x=filtered_df['index_readiness_score'].mean(), line_dash="dash", line_color="green", annotation_text=f"Mean ({filtered_df['index_readiness_score'].mean():.1f})")

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================

st.caption("RAG Index-Readiness Pipeline | MGTA 452 Project | Complete with Anomaly Detection")
