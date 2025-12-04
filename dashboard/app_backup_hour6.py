"""
RAG Index-Readiness Dashboard - Enhanced with Cost Calculator & Exports
Part 2: Production Features
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch

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
    
    # Title
    title = Paragraph("<b>RAG Index-Readiness Quality Report</b>", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
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
    
    # Quality Distribution Table
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
    
    # Recommendations
    elements.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
    
    recommendations = f"""
    1. <b>Index {(df['index_ready']).sum():,} reviews immediately</b> - These meet quality standards<br/>
    2. <b>Review {(df['has_pii']).sum():,} reviews with PII</b> - Consider redaction before indexing<br/>
    3. <b>Monitor chunk quality</b> - {(df['chunk_quality_flag'] == 'too_short').sum():,} reviews are too short<br/>
    4. <b>Estimated Cost Savings</b> - Filtering saves ~${(len(df) - (df['index_ready']).sum()) * 0.25 / 1000 * 12:,.0f}/year in vector DB costs<br/>
    """
    elements.append(Paragraph(recommendations, styles['Normal']))
    
    # Build PDF
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
    
    # Parse date column
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Parse PII types (JSON string to list)
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
# SIDEBAR - FILTERS
# ============================================================================

st.sidebar.header("🔍 Filters")

# Filter 1: Date Range
st.sidebar.subheader("Date Range")
min_date = df['date'].min().date()
max_date = df['date'].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Handle single date selection
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

# Filter 2: Star Rating
st.sidebar.subheader("Star Rating")
star_options = sorted(df['stars'].unique())
selected_stars = st.sidebar.multiselect(
    "Select star ratings",
    options=star_options,
    default=star_options
)

# Filter 3: Quality Tier
st.sidebar.subheader("Quality Tier")
quality_options = ['All', 'Index Ready', 'Needs Review', 'Reject']
selected_quality = st.sidebar.selectbox(
    "Select quality tier",
    options=quality_options,
    index=0
)

# Apply filters
filtered_df = df.copy()

# Date filter
filtered_df = filtered_df[
    (filtered_df['date'].dt.date >= start_date) & 
    (filtered_df['date'].dt.date <= end_date)
]

# Star rating filter
if selected_stars:
    filtered_df = filtered_df[filtered_df['stars'].isin(selected_stars)]

# Quality tier filter
if selected_quality != 'All':
    if selected_quality == 'Index Ready':
        filtered_df = filtered_df[filtered_df['recommendation'] == 'index']
    elif selected_quality == 'Needs Review':
        filtered_df = filtered_df[filtered_df['recommendation'] == 'review']
    elif selected_quality == 'Reject':
        filtered_df = filtered_df[filtered_df['recommendation'] == 'reject']

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(f"**Filtered:** {len(filtered_df):,} / {len(df):,} reviews ({len(filtered_df)/len(df)*100:.1f}%)")

# ============================================================================
# 🆕 COST CALCULATOR WIDGET
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.header("💰 Cost Impact Calculator")

# Input parameters
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

# Calculate savings
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
# MAIN DASHBOARD - HEADER
# ============================================================================

st.title("📊 RAG Index-Readiness Dashboard")
st.markdown("### Yelp Reviews Data Quality Analysis")

# 🆕 Export buttons at top
col1, col2, col3 = st.columns(3)

with col1:
    # Export 1: Index-ready reviews
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
    # Export 2: PII reviews
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
    # Export 3: PDF Report
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
# KPI METRICS (4 KEY METRICS)
# ============================================================================

st.header("🎯 Key Performance Indicators")

# Calculate KPIs
total_reviews = len(filtered_df)
avg_score = filtered_df['index_readiness_score'].mean()
duplicate_rate = (filtered_df['is_duplicate'].sum() / total_reviews * 100) if total_reviews > 0 else 0
pii_rate = (filtered_df['has_pii'].sum() / total_reviews * 100) if total_reviews > 0 else 0
index_ready_rate = (filtered_df['index_ready'].sum() / total_reviews * 100) if total_reviews > 0 else 0

# Display KPIs in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Reviews",
        value=f"{total_reviews:,}",
        delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
    )

with col2:
    st.metric(
        label="Avg Quality Score",
        value=f"{avg_score:.1f}/100",
        delta=f"{avg_score - df['index_readiness_score'].mean():.1f}" if len(filtered_df) != len(df) else None
    )

with col3:
    st.metric(
        label="Duplicate Rate",
        value=f"{duplicate_rate:.1f}%",
        delta=f"{duplicate_rate - (df['is_duplicate'].sum() / len(df) * 100):.1f}%" if len(filtered_df) != len(df) else None,
        delta_color="inverse"
    )

with col4:
    st.metric(
        label="PII Risk",
        value=f"{pii_rate:.1f}%",
        delta=f"{pii_rate - (df['has_pii'].sum() / len(df) * 100):.1f}%" if len(filtered_df) != len(df) else None,
        delta_color="inverse"
    )

st.markdown("---")

# ============================================================================
# QUALITY DISTRIBUTION
# ============================================================================

st.header("📈 Quality Distribution")

col1, col2 = st.columns([2, 1])

with col1:
    quality_counts = filtered_df['recommendation'].value_counts().reset_index()
    quality_counts.columns = ['Recommendation', 'Count']
    
    quality_map = {'index': 'Index Ready', 'review': 'Needs Review', 'reject': 'Reject'}
    quality_counts['Recommendation'] = quality_counts['Recommendation'].map(quality_map)
    
    fig = px.bar(
        quality_counts,
        x='Recommendation',
        y='Count',
        title='Reviews by Quality Tier',
        color='Recommendation',
        color_discrete_map={
            'Index Ready': '#10b981',
            'Needs Review': '#f59e0b',
            'Reject': '#ef4444'
        }
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Quality Tier",
        yaxis_title="Number of Reviews",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Summary Statistics:**")
    
    stats_df = pd.DataFrame({
        'Metric': [
            'Index Ready',
            'Needs Review',
            'Reject',
            'Avg Score',
            'Median Score'
        ],
        'Value': [
            f"{(filtered_df['recommendation'] == 'index').sum():,} ({index_ready_rate:.1f}%)",
            f"{(filtered_df['recommendation'] == 'review').sum():,} ({(filtered_df['recommendation'] == 'review').sum()/total_reviews*100:.1f}%)",
            f"{(filtered_df['recommendation'] == 'reject').sum():,} ({(filtered_df['recommendation'] == 'reject').sum()/total_reviews*100:.1f}%)",
            f"{avg_score:.1f}/100",
            f"{filtered_df['index_readiness_score'].median():.1f}/100"
        ]
    })
    
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

st.markdown("---")

# ============================================================================
# SCORE DISTRIBUTION HISTOGRAM
# ============================================================================

st.header("📊 Score Distribution")

fig = px.histogram(
    filtered_df,
    x='index_readiness_score',
    nbins=20,
    title='Distribution of Index-Readiness Scores',
    color_discrete_sequence=['#3b82f6']
)

fig.update_layout(
    xaxis_title="Index-Readiness Score",
    yaxis_title="Number of Reviews",
    height=400,
    showlegend=False
)

fig.add_vline(x=70, line_dash="dash", line_color="green", annotation_text="Index Ready (70+)")
fig.add_vline(x=50, line_dash="dash", line_color="orange", annotation_text="Needs Review (50+)")

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================================
# CHUNK QUALITY BREAKDOWN
# ============================================================================

st.header("🔧 Chunk Quality Analysis")

col1, col2 = st.columns(2)

with col1:
    chunk_counts = filtered_df['chunk_quality_flag'].value_counts().reset_index()
    chunk_counts.columns = ['Quality', 'Count']
    
    fig = px.pie(
        chunk_counts,
        values='Count',
        names='Quality',
        title='Chunk Quality Distribution',
        color='Quality',
        color_discrete_map={
            'optimal': '#10b981',
            'too_short': '#f59e0b',
            'too_long': '#ef4444'
        }
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    chunk_scores = filtered_df.groupby('chunk_quality_flag')['index_readiness_score'].mean().reset_index()
    chunk_scores.columns = ['Chunk Quality', 'Avg Score']
    
    fig = px.bar(
        chunk_scores,
        x='Chunk Quality',
        y='Avg Score',
        title='Avg Quality Score by Chunk Type',
        color='Chunk Quality',
        color_discrete_map={
            'optimal': '#10b981',
            'too_short': '#f59e0b',
            'too_long': '#ef4444'
        }
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        yaxis_range=[0, 100]
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ============================================================================
# PII ANALYSIS
# ============================================================================

st.header("🔒 PII Detection Analysis")

col1, col2 = st.columns(2)

with col1:
    pii_counts = filtered_df['has_pii'].value_counts().reset_index()
    pii_counts.columns = ['Has PII', 'Count']
    pii_counts['Has PII'] = pii_counts['Has PII'].map({True: 'Contains PII', False: 'No PII'})
    
    fig = px.pie(
        pii_counts,
        values='Count',
        names='Has PII',
        title='PII Detection Results',
        color='Has PII',
        color_discrete_map={
            'No PII': '#10b981',
            'Contains PII': '#ef4444'
        }
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**PII Types Detected:**")
    
    pii_type_counts = {}
    for pii_list in filtered_df[filtered_df['has_pii']]['pii_types_list']:
        for pii_type in pii_list:
            pii_type_counts[pii_type] = pii_type_counts.get(pii_type, 0) + 1
    
    if pii_type_counts:
        pii_df = pd.DataFrame({
            'PII Type': list(pii_type_counts.keys()),
            'Count': list(pii_type_counts.values())
        }).sort_values('Count', ascending=False)
        
        fig = px.bar(
            pii_df,
            x='PII Type',
            y='Count',
            title='PII Types Breakdown',
            color_discrete_sequence=['#ef4444']
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No PII detected in filtered data.")

st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("RAG Index-Readiness Pipeline | MGTA 452 Project | Enhanced with Cost Calculator & Export Features")
