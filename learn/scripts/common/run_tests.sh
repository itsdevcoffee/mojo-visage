#!/bin/bash
# Run all tests in the tests/ directory

echo "🧪 Running all tests..."
echo ""

# Find all test files recursively
find tests -name "test_*.mojo" -type f | sort | while read test_file; do
    echo "Running: $test_file"
    mojo "$test_file"
    echo ""
done

echo "✅ All test files executed!"
