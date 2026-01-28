#!/bin/bash
# Run all code quality checks
# Usage: ./scripts/quality.sh [format]

set -e
cd "$(dirname "$0")/.."

case "$1" in
    format)
        echo "=== Formatting Code ==="
        uv run black .
        ;;
    *)
        echo "=== Code Quality Checks ==="
        echo "--- Black Format Check ---"
        uv run black --check .
        echo "All checks passed!"
        ;;
esac
