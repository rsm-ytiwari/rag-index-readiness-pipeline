#!/bin/bash
echo "==================================="
echo "FINDING YELP DATASET"
echo "==================================="

echo -e "\n1. Current directory:"
pwd

echo -e "\n2. Listing parent directory:"
ls -la .. | head -20

echo -e "\n3. Searching for yelp files:"
find /home/jovyan -name "*yelp*.json" -type f 2>/dev/null

echo -e "\n4. Checking common data locations:"
for dir in \
    "/home/jovyan/data" \
    "/home/jovyan/MSBA/data" \
    "/home/jovyan/MSBA/02_PROJECTS/data" \
    "/home/jovyan/work/data" \
    "$HOME/data"
do
    if [ -d "$dir" ]; then
        echo "  Found: $dir"
        ls -lh "$dir" 2>/dev/null | grep -i yelp
    fi
done

echo -e "\n5. Docker volume mounts (if any):"
df -h | grep -E "data|Data|DATA"

echo "==================================="
