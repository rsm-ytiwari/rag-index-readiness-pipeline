"""
RAG Index-Readiness Dashboard
Streamlit app for visualizing data quality metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

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
# DATA LOADING WITH CACHING
# ============================================================================

@st.cache_data
def load_data():
    """Load Gold Parquet with caching for performance."""
    data_path = Path("../data/gold/reviews_featured.parquet")
    
    if not data_path.exists():
        # Try alternative path (if running from project root)
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
# MAIN DASHBOARD - HEADER
# ============================================================================

st.title("📊 RAG Index-Readiness Dashboard")
st.markdown("### Yelp Reviews Data Quality Analysis")
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
        delta_color="inverse"  # Lower is better
    )

with col4:
    st.metric(
        label="PII Risk",
        value=f"{pii_rate:.1f}%",
        delta=f"{pii_rate - (df['has_pii'].sum() / len(df) * 100):.1f}%" if len(filtered_df) != len(df) else None,
        delta_color="inverse"  # Lower is better
    )

st.markdown("---")

# ============================================================================
# QUALITY DISTRIBUTION
# ============================================================================

st.header("📈 Quality Distribution")

col1, col2 = st.columns([2, 1])

with col1:
    # Bar chart: Reviews by recommendation tier
    quality_counts = filtered_df['recommendation'].value_counts().reset_index()
    quality_counts.columns = ['Recommendation', 'Count']
    
    # Map to proper labels
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
    # Stats table
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

# Add vertical lines for thresholds
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
    # Pie chart: Chunk quality distribution
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
    # Bar chart: Avg score by chunk quality
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
    # Pie chart: PII presence
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
    # PII type breakdown
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
st.caption("RAG Index-Readiness Pipeline | MGTA 452 Project | Data: Yelp Academic Dataset")
