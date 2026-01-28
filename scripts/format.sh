#!/bin/bash
# Format code using black
# Usage: ./scripts/format.sh [--check]

set -e
cd "$(dirname "$0")/.."

if [ "$1" == "--check" ]; then
    echo "Checking format..."
    uv run black --check .
else
    echo "Formatting code..."
    uv run black .
fi
