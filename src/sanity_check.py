"""
Sanity check script - Run this to verify setup before running 01_ingest.py
"""
from pathlib import Path
import sys

print("="*70)
print("SANITY CHECK - Verifying Environment")
print("="*70)

checks_passed = 0
checks_failed = 0

# Check 1: Project structure
print("\n1. Checking project structure...")
required_dirs = ['src', 'data', 'data/bronze', 'data/silver', 'data/gold', 'data/Raw', 'queries', 'dashboard']
for dir_name in required_dirs:
    if Path(dir_name).exists():
        print(f"   ✅ {dir_name}/")
        checks_passed += 1
    else:
        print(f"   ❌ {dir_name}/ MISSING")
        checks_failed += 1

# Check 2: Python packages
print("\n2. Checking Python packages...")
try:
    import pandas as pd
    print(f"   ✅ pandas {pd.__version__}")
    checks_passed += 1
except ImportError:
    print("   ❌ pandas NOT INSTALLED")
    checks_failed += 1

try:
    import pyarrow as pa
    print(f"   ✅ pyarrow {pa.__version__}")
    checks_passed += 1
except ImportError:
    print("   ❌ pyarrow NOT INSTALLED")
    checks_failed += 1

# Check 3: Input file (CORRECTED PATH)
print("\n3. Checking input data file...")
input_file = Path("data/Raw/yelp_academic_dataset_review.json")
if input_file.exists():
    size_gb = input_file.stat().st_size / (1024**3)
    print(f"   ✅ Input file exists ({size_gb:.2f} GB)")
    checks_passed += 1
else:
    print(f"   ❌ Input file NOT FOUND: {input_file}")
    checks_failed += 1

# Check 4: Output directory writable
print("\n4. Checking write permissions...")
test_file = Path("data/bronze/.test_write")
try:
    test_file.touch()
    test_file.unlink()
    print("   ✅ Bronze directory is writable")
    checks_passed += 1
except Exception as e:
    print(f"   ❌ Cannot write to bronze directory: {e}")
    checks_failed += 1

# Summary
print(f"\n{'='*70}")
print(f"SUMMARY: {checks_passed} passed, {checks_failed} failed")
print(f"{'='*70}")

if checks_failed > 0:
    print("❌ SANITY CHECK FAILED - Fix issues before proceeding")
    sys.exit(1)
else:
    print("✅ SANITY CHECK PASSED - Ready to run 01_ingest.py")
    sys.exit(0)
