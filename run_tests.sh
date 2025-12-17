#!/bin/bash

# Test Runner Script for MCP Research Assistant
# This script provides convenient commands for running tests

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}MCP Research Assistant - Test Runner${NC}"
echo "======================================"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found${NC}"
    echo "Install test dependencies with: pip install -r tests/requirements-test.txt"
    exit 1
fi

# Parse command line argument
TEST_TYPE=${1:-all}

case $TEST_TYPE in
    "all")
        echo -e "${BLUE}Running all tests with coverage...${NC}"
        pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
        ;;
    "unit")
        echo -e "${BLUE}Running unit tests only...${NC}"
        pytest tests/unit/ -v --cov=src --cov-report=term-missing
        ;;
    "integration")
        echo -e "${BLUE}Running integration tests only...${NC}"
        pytest tests/integration/ -v --cov=src --cov-report=term-missing
        ;;
    "fast")
        echo -e "${BLUE}Running fast tests (excluding slow)...${NC}"
        pytest tests/ -m "not slow" -v
        ;;
    "coverage")
        echo -e "${BLUE}Generating detailed coverage report...${NC}"
        pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    "quick")
        echo -e "${BLUE}Quick test run (no coverage)...${NC}"
        pytest tests/ -v --tb=short
        ;;
    "failed")
        echo -e "${BLUE}Re-running last failed tests...${NC}"
        pytest tests/ --lf -v
        ;;
    "debug")
        echo -e "${BLUE}Running tests in debug mode...${NC}"
        pytest tests/ -v -s --tb=long
        ;;
    "parallel")
        echo -e "${BLUE}Running tests in parallel...${NC}"
        if ! command -v pytest-xdist &> /dev/null; then
            echo -e "${RED}pytest-xdist not installed. Install with: pip install pytest-xdist${NC}"
            exit 1
        fi
        pytest tests/ -n auto -v
        ;;
    "help")
        echo "Usage: ./run_tests.sh [option]"
        echo ""
        echo "Options:"
        echo "  all         - Run all tests with coverage (default)"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests only"
        echo "  fast        - Run fast tests (exclude slow)"
        echo "  coverage    - Generate detailed coverage report"
        echo "  quick       - Quick run without coverage"
        echo "  failed      - Re-run last failed tests"
        echo "  debug       - Run with debug output"
        echo "  parallel    - Run tests in parallel (requires pytest-xdist)"
        echo "  help        - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh              # Run all tests"
        echo "  ./run_tests.sh unit         # Unit tests only"
        echo "  ./run_tests.sh coverage     # Generate coverage report"
        ;;
    *)
        echo -e "${RED}Unknown option: $TEST_TYPE${NC}"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Tests completed successfully${NC}"
    
    # Show coverage summary if available
    if [ -f .coverage ]; then
        echo ""
        echo "Coverage summary:"
        coverage report --skip-empty
    fi
else
    echo ""
    echo -e "${RED}✗ Tests failed${NC}"
    exit 1
fi
