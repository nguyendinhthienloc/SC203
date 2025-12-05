#!/bin/bash
# Install dependencies for GECS integration

echo "Installing GECS dependencies..."

# Core dependencies (if not already installed)
pip install openai rouge scikit-learn

echo ""
echo "✓ GECS dependencies installed!"
echo ""
echo "To verify installation:"
echo "  python -c 'import openai; from rouge import Rouge; print(\"✓ Ready\")'"
echo ""
echo "To run GECS analysis:"
echo "  python run_with_gecs.py"
