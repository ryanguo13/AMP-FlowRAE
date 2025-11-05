#!/bin/bash
# Quick environment setup using uv

set -e

echo "🔧 Setting up AMP-FlowRAE environment with uv..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed. Please restart your shell and run this script again."
    exit 0
fi

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    uv venv
fi

# Activate venv
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
uv pip install -e .

# Install ESM package
echo "📥 Installing ESM package..."
uv pip install fair-esm biopython

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run Phase 1:"
echo "  bash scripts/run_phase1.sh"



