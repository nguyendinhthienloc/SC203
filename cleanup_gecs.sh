#!/bin/bash
# GECS Refactoring Script
# This script cleans up redundant GECS files

echo "GECS Refactoring - Cleanup Script"
echo "=================================="
echo ""

# Files to delete
FILES_TO_DELETE=(
    "gecs.py"
    "gecs_demo.py"
    "run_gecs_analysis.py"
    "scripts/convert_csv_to_json_gecs.py"
    "scripts/install_gecs_deps.py"
    "GECS_COMPLETE.md"
    "GECS_README.md"
    "GECS_CHECKLIST.md"
    "docs/GECS_INTEGRATION_SUMMARY.md"
    "docs/GECS_INTEGRATION.md"
    "docs/GECS_ANALYSIS.md"
    "install_gecs.sh"
)

echo "The following files will be deleted:"
for file in "${FILES_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done

echo ""
echo "Files to keep:"
echo "  - src/gec_score.py (core module)"
echo "  - run_with_gecs.py (main runner)"
echo "  - docs/GECS_SUMMARY.md (consolidated docs)"
echo ""

read -p "Proceed with deletion? [y/N]: " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for file in "${FILES_TO_DELETE[@]}"; do
        if [ -f "$file" ]; then
            rm "$file"
            echo "✓ Deleted: $file"
        fi
    done
    echo ""
    echo "✅ Cleanup complete!"
    echo ""
    echo "Remaining GECS files:"
    echo "  - src/gec_score.py"
    echo "  - run_with_gecs.py"
    echo "  - docs/GECS_SUMMARY.md"
    echo "  - data/HC3/hc3_sample.json"
else
    echo "Aborted."
fi
