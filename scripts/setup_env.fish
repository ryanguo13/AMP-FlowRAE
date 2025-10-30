#!/usr/bin/env fish
# Quick environment setup using uv (Fish shell version)

echo "🔧 Setting up AMP-FlowRAE environment with uv..."
echo ""

# Check if uv is installed
if not command -v uv &>/dev/null
    echo "❌ uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed. Please restart your shell and run this script again."
    exit 0
end

# Create venv if not exists
if not test -d .venv
    echo "📦 Creating virtual environment..."
    uv venv
end

# Activate venv
echo "🔌 Activating virtual environment..."
source .venv/bin/activate.fish

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
echo "  source .venv/bin/activate.fish"
echo ""
echo "Then run Phase 1:"
echo "  fish scripts/run_phase1.fish"

