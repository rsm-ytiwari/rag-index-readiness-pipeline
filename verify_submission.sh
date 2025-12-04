#!/bin/bash

echo "=========================================="
echo "PROJECT VERIFICATION SCRIPT"
echo "=========================================="

# Check file structure
echo -e "\n1. Checking file structure..."
if [ -d "src" ] && [ -d "api" ] && [ -d "dashboard" ]; then
    echo "   ✅ All directories present"
else
    echo "   ❌ Missing directories"
fi

# Check data files
echo -e "\n2. Checking data files..."
if [ -f "data/gold/reviews_featured.parquet" ]; then
    echo "   ✅ Gold parquet exists"
else
    echo "   ❌ Gold parquet missing"
fi

# Check documentation
echo -e "\n3. Checking documentation..."
if [ -f "README.md" ] && [ -f "PROJECT_SUMMARY.md" ]; then
    echo "   ✅ Documentation complete"
else
    echo "   ❌ Documentation missing"
fi

# Count lines of code
echo -e "\n4. Code statistics..."
echo "   Python files: $(find . -name "*.py" | wc -l)"
echo "   Total lines: $(find . -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $1}')"

# Check dependencies
echo -e "\n5. Checking requirements..."
if [ -f "requirements.txt" ]; then
    echo "   ✅ requirements.txt exists"
    echo "   Packages: $(cat requirements.txt | grep -v "^#" | grep -v "^$" | wc -l)"
else
    echo "   ❌ requirements.txt missing"
fi

echo -e "\n=========================================="
echo "✅ VERIFICATION COMPLETE"
echo "=========================================="
