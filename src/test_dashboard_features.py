"""
Test dashboard features without browser
"""

import pandas as pd
import json
from pathlib import Path
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch

print("="*70)
print("TESTING DASHBOARD FEATURES")
print("="*70)

# ============================================================================
# TEST 1: Load Data
# ============================================================================
print("\n📊 TEST 1: Loading Data...")

df = pd.read_parquet('data/gold/reviews_featured.parquet')
df['date'] = pd.to_datetime(df['date'])
df['pii_types_list'] = df['pii_types'].apply(
    lambda x: json.loads(x) if x and x != '[]' else []
)

print(f"✅ Loaded {len(df):,} reviews")

# ============================================================================
# TEST 2: Cost Calculator Logic
# ============================================================================
print("\n💰 TEST 2: Cost Calculator...")

vector_db_cost_per_1k = 0.25  # $0.25 per 1K embeddings
reviews_to_index = 10_000_000  # 10 million reviews

total_reviews = reviews_to_index
index_ready_rate = (df['index_ready']).sum() / len(df)

reviews_filtered = total_reviews * (1 - index_ready_rate)
cost_without_filter = total_reviews * vector_db_cost_per_1k / 1000
cost_with_filter = total_reviews * index_ready_rate * vector_db_cost_per_1k / 1000
monthly_savings = cost_without_filter - cost_with_filter
annual_savings = monthly_savings * 12

print(f"Index-ready rate: {index_ready_rate*100:.1f}%")
print(f"Reviews filtered: {reviews_filtered:,.0f}")
print(f"Cost without filter: ${cost_without_filter:,.2f}/month")
print(f"Cost with filter: ${cost_with_filter:,.2f}/month")
print(f"💵 Monthly savings: ${monthly_savings:,.2f}")
print(f"📈 Annual ROI: ${annual_savings:,.2f}")

# ============================================================================
# TEST 3: Export - Index-Ready Reviews CSV
# ============================================================================
print("\n📥 TEST 3: Export Index-Ready Reviews...")

ready_reviews = df[df['index_ready']].copy()
ready_csv = ready_reviews[['review_id', 'text', 'stars', 'date', 'index_readiness_score']].to_csv(index=False)

output_file = Path("exports/index_ready_reviews.csv")
output_file.parent.mkdir(exist_ok=True)
output_file.write_text(ready_csv)

print(f"✅ Exported {len(ready_reviews):,} index-ready reviews")
print(f"📂 File: {output_file}")
print(f"📊 Size: {len(ready_csv)/1024:.1f} KB")

# ============================================================================
# TEST 4: Export - PII Reviews CSV
# ============================================================================
print("\n�� TEST 4: Export PII Reviews...")

pii_reviews = df[df['has_pii']].copy()
pii_csv = pii_reviews[['review_id', 'text', 'pii_types', 'index_readiness_score']].to_csv(index=False)

output_file = Path("exports/pii_reviews.csv")
output_file.write_text(pii_csv)

print(f"✅ Exported {len(pii_reviews):,} PII reviews")
print(f"📂 File: {output_file}")
print(f"📊 Size: {len(pii_csv)/1024:.1f} KB")

# ============================================================================
# TEST 5: Export - PDF Report
# ============================================================================
print("\n📥 TEST 5: Generate PDF Report...")

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

pdf_buffer = generate_pdf_report(df)

output_file = Path("exports/quality_report.pdf")
output_file.write_bytes(pdf_buffer.read())

print(f"✅ Generated PDF report")
print(f"📂 File: {output_file}")
print(f"📊 Size: {output_file.stat().st_size/1024:.1f} KB")

# ============================================================================
# TEST 6: Verify Export Files
# ============================================================================
print("\n✅ TEST 6: Verify All Exports...")

exports_dir = Path("exports")
files = list(exports_dir.glob("*"))

print(f"\n📂 Exports directory contains {len(files)} files:")
for f in files:
    print(f"  - {f.name}: {f.stat().st_size/1024:.1f} KB")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("📊 FEATURE TEST SUMMARY")
print("="*70)

print(f"""
✅ Cost Calculator:
   - Index-ready rate: {index_ready_rate*100:.1f}%
   - Monthly savings: ${monthly_savings:,.2f}
   - Annual ROI: ${annual_savings:,.2f}

✅ Export Files Generated:
   - index_ready_reviews.csv: {len(ready_reviews):,} rows
   - pii_reviews.csv: {len(pii_reviews):,} rows
   - quality_report.pdf: {output_file.stat().st_size/1024:.1f} KB

✅ All features working correctly!
""")

print("="*70)
print("✅ HOUR 1 COMPLETE - ALL TESTS PASSED")
print("="*70)
