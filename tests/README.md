# Test Suite Documentation

## Overview

This test suite provides comprehensive coverage for the MCP Research Assistant, including:
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test complete workflows and API endpoints
- **Coverage**: Measure code coverage to ensure quality

## Test Structure

```
tests/
├── conftest.py                      # Shared fixtures and configuration
├── unit/                            # Unit tests
│   ├── test_mcp_retriever.py       # MCPRetrieverAgent tests
│   ├── test_summarizer.py          # SummarizerAgent tests
│   ├── test_verifier.py            # ThoroughMCPVerifier tests
│   ├── test_multi_mode_assistant.py # MultiModeResearchAssistant tests
│   ├── test_tool_executors.py      # MCPToolExecutor tests
│   └── test_benchmark.py           # Benchmark module tests
└── integration/                     # Integration tests
    ├── test_api.py                 # API endpoint tests
    └── test_workflows.py           # End-to-end workflow tests
```

## Installation

### Install Test Dependencies

```bash
# Install pytest and related packages
pip install pytest pytest-asyncio pytest-cov pytest-mock

# Or use the project requirements
pip install -r requirements.txt
```

### Required Packages

- `pytest>=7.0.0` - Test framework
- `pytest-asyncio>=0.21.0` - Async test support
- `pytest-cov>=4.0.0` - Coverage reporting
- `pytest-mock>=3.10.0` - Mocking utilities

## Running Tests

### Run All Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run with detailed output
pytest tests/ -v -s
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_mcp_retriever.py -v

# Specific test function
pytest tests/unit/test_mcp_retriever.py::TestMCPRetrieverAgent::test_process_valid_query_arxiv -v
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit -v

# Run only integration tests
pytest -m integration -v

# Run only slow tests
pytest -m slow -v

# Run tests that require API keys
pytest -m requires_api -v

# Exclude slow tests
pytest -m "not slow" -v
```

### Coverage Reports

```bash
# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html

# View coverage in terminal
pytest tests/ --cov=src --cov-report=term-missing

# Generate XML report (for CI/CD)
pytest tests/ --cov=src --cov-report=xml

# Open HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Test Configuration

### pytest.ini

The `pytest.ini` file contains:
- Test discovery patterns
- Coverage configuration
- Timeout settings
- Custom markers

### .coveragerc

Coverage configuration includes:
- Source directories
- Files to omit
- Reporting options
- HTML output directory

### Environment Variables

Set these for integration tests:

```bash
export OPENAI_API_KEY="your_key_here"
export TAVILY_API_KEY="your_key_here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

## Test Categories

### Unit Tests (tests/unit/)

**Purpose**: Test individual components in isolation with mocked dependencies

**Coverage**:
- MCPRetrieverAgent: 15+ tests
- SummarizerAgent: 13+ tests
- ThoroughMCPVerifier: 13+ tests
- MultiModeResearchAssistant: 15+ tests
- MCPToolExecutor: 15+ tests
- Benchmark module: 15+ tests

**Total**: ~86+ unit tests

### Integration Tests (tests/integration/)

**Purpose**: Test complete workflows and API endpoints

**Coverage**:
- API endpoints: 15+ tests
- Complete workflows: 9+ tests

**Total**: ~24+ integration tests

## Test Fixtures

Common fixtures available in `conftest.py`:

- `test_api_keys`: Test API keys
- `sample_query`: Sample research query
- `sample_papers`: Mock paper data
- `sample_web_results`: Mock web search results
- `sample_summary`: Mock summary text
- `mock_openai_client`: Mocked OpenAI client
- `mock_embedder`: Mocked SentenceTransformer
- `mock_tool_executor`: Mocked tool executor
- `mcp_retriever`: MCPRetrieverAgent with mocked dependencies
- `summarizer_agent`: SummarizerAgent with mocked dependencies
- `verifier_agent`: ThoroughMCPVerifier with mocked dependencies
- `multi_mode_assistant`: MultiModeResearchAssistant with mocked dependencies

## Writing New Tests

### Unit Test Template

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.unit
class TestYourComponent:
    """Test suite for YourComponent"""
    
    @pytest.mark.asyncio
    async def test_your_function(self, mock_dependency):
        """Test description"""
        # Arrange
        component = YourComponent()
        
        # Act
        result = await component.your_function()
        
        # Assert
        assert result is not None
```

### Integration Test Template

```python
import pytest

@pytest.mark.integration
class TestYourWorkflow:
    """Integration tests for your workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, test_api_keys):
        """Test description"""
        # Setup
        # ...
        
        # Execute workflow
        result = await execute_workflow()
        
        # Verify
        assert result["success"] is True
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    
    - name: Run tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        TAVILY_API_KEY: ${{ secrets.TAVILY_API_KEY }}
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Troubleshooting

### Common Issues

**Issue**: Import errors
```bash
# Solution: Install package in editable mode
pip install -e .
```

**Issue**: Async tests not running
```bash
# Solution: Install pytest-asyncio
pip install pytest-asyncio
```

**Issue**: Coverage not including src/
```bash
# Solution: Run from project root
cd /path/to/MCP_Deep_Research_Bot-main
pytest tests/ --cov=src
```

**Issue**: Tests timeout
```bash
# Solution: Increase timeout in pytest.ini or use marker
pytest tests/ --timeout=600
```

### Debug Tests

```bash
# Run with print statements visible
pytest tests/ -v -s

# Run with Python debugger
pytest tests/ --pdb

# Run last failed tests only
pytest tests/ --lf

# Run with verbose error output
pytest tests/ -vv
```

## Test Metrics

### Current Coverage

Target: **>70% code coverage**

Run to check current coverage:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Counts

- **Total Tests**: ~110+
- **Unit Tests**: ~86+
- **Integration Tests**: ~24+

### Execution Time

- **Unit Tests**: ~10-30 seconds
- **Integration Tests**: ~30-60 seconds
- **Total Suite**: ~1-2 minutes

## Best Practices

1. **Use Descriptive Names**: Test names should describe what they test
2. **One Assertion Per Test**: Keep tests focused
3. **Use Fixtures**: Reuse common setup via fixtures
4. **Mock External Calls**: Don't hit real APIs in tests
5. **Test Edge Cases**: Empty inputs, errors, boundary conditions
6. **Keep Tests Fast**: Mock heavy operations
7. **Make Tests Independent**: Each test should run in isolation
8. **Document Tests**: Add docstrings explaining test purpose

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure >70% coverage for new code
3. Run full test suite before committing
4. Update this README if adding new test categories

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio documentation](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
