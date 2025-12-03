"""
Test all DuckDB queries from analysis.sql
"""

import duckdb
from pathlib import Path
import time

print("="*70)
print("TESTING DUCKDB QUERIES")
print("="*70)

# Connect to DuckDB
con = duckdb.connect()

# Read SQL file
sql_file = Path("queries/analysis.sql")
print(f"\n📂 Loading queries from: {sql_file}")

with open(sql_file, 'r') as f:
    sql_content = f.read()

# Split queries (simple split on double newline + SELECT)
queries = []
current_query = []
query_name = None

for line in sql_content.split('\n'):
    if line.startswith('-- QUERY'):
        if current_query and query_name:
            queries.append((query_name, '\n'.join(current_query)))
        query_name = line.replace('--', '').strip()
        current_query = []
    elif line.strip() and not line.startswith('--'):
        current_query.append(line)

# Add last query
if current_query and query_name:
    queries.append((query_name, '\n'.join(current_query)))

print(f"✅ Loaded {len(queries)} queries\n")

# Test each query
for i, (name, query) in enumerate(queries, 1):
    print(f"{'─'*70}")
    print(f"QUERY {i}: {name}")
    print(f"{'─'*70}")
    
    try:
        start_time = time.time()
        result = con.execute(query).fetchdf()
        elapsed = time.time() - start_time
        
        print(f"✅ Success ({elapsed:.2f}s)")
        print(f"📊 Result: {len(result)} rows × {len(result.columns)} columns")
        print(f"\n{result.to_string()}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}\n")

print("="*70)
print("✅ QUERY TESTING COMPLETE")
print("="*70)
