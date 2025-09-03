# Development Guide

Guide for developers working on the Quant Trading System.

## 🚀 Quick Setup

### Prerequisites
- Python 3.12+
- Poetry
- Git

### Installation
```bash
git clone https://github.com/LouisLetcher/quant-system.git
cd quant-system
poetry install --with dev
poetry shell
pre-commit install
```

## 🧪 Testing

### Running Tests
```bash
# All tests with coverage
pytest

# Unit tests only
pytest -m "not integration"

# Integration tests
pytest -m "integration"

# Specific test file
pytest tests/test_data_manager.py

# Parallel execution
pytest -n auto
```

### Database for Tests
- By default, unit tests and CI use a lightweight SQLite database to avoid any external Postgres dependency.
- The unified DB models auto-detect CI/pytest and prefer SQLite when any of these env vars are present: `CI`, `PYTEST_CURRENT_TEST`, or `TESTING`.
- You can force this behavior explicitly by setting:
  - `UNIFIED_MODELS_SQLITE=1`
  - Optionally, also set `DATABASE_URL=sqlite:///quant_unified_test.db` for consistency.

Examples:
```bash
# Local: force SQLite for tests
export UNIFIED_MODELS_SQLITE=1
export DATABASE_URL=sqlite:///quant_unified_test.db
pytest
```

### Test Structure
- `tests/test_*.py` - Unit tests
- `tests/test_integration.py` - Integration tests
- `tests/conftest.py` - Shared fixtures and configuration

## 🔍 Code Quality

### Formatting and Linting
```bash
ruff check .     # Lint and format code
ruff format .    # Format code (alternative)
mypy src/        # Type checking
```

### Security Checks
```bash
bandit -r src/   # Security linting
safety check     # Dependency vulnerabilities
```

### Pre-commit Hooks
Pre-commit hooks run automatically on git commit:
- Code formatting and linting (Ruff)
- Type checking (MyPy)
- Security scanning (Bandit)

## 📁 Project Structure

```
src/
├── ai/                   # AI recommendation system
├── backtesting_engine/   # Strategies submodule (quant-strategies repo)
│   └── algorithms/python/ # Python strategy implementations (40+ strategies)
├── cli/                  # Command-line interface
├── core/                 # Core system components
│   ├── data_manager.py   # Data fetching and management
│   ├── direct_backtest.py # Direct backtesting library integration
│   └── portfolio_manager.py # Portfolio management
├── database/             # Database models and operations
├── portfolio/            # Portfolio optimization
├── reporting/            # Report generation
└── utils/                # Utility functions

tests/
├── test_*.py            # Unit tests
├── test_integration.py  # Integration tests
└── conftest.py          # Test configuration

config/
└── collections/         # Asset collections
```

## 🔧 Development Commands

### Building
```bash
poetry build        # Build package
docker build .      # Build Docker image
```

### Running Services
```bash
# CLI discovery
python -m src.cli.unified_cli --help

# Docker development (compose v2)
docker compose up -d postgres pgadmin
docker compose build quant
docker compose run --rm quant python -m src.cli.unified_cli --help
```

## 📝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes
4. **Add** tests for new functionality
5. **Ensure** all tests pass: `pytest`
6. **Commit** your changes: `git commit -m 'Add amazing feature'`
7. **Push** to the branch: `git push origin feature/amazing-feature`
8. **Open** a Pull Request

### Code Style Guidelines
- **Line length**: 88 characters (Black default)
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style for all modules, classes, and functions
- **Tests**: Required for all new functionality

### Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(data): add Alpha Vantage data source`
- `fix(portfolio): correct position sizing calculation`
- `docs(readme): update installation instructions`

## 🔍 Debugging

### Environment Variables
```bash
export LOG_LEVEL=DEBUG
export TESTING=true
```

### Common Issues
1. **Import errors**: Ensure virtual environment is activated
2. **API failures**: Check API keys in `.env` file
3. **Permission errors**: Check file permissions for cache/exports directories

### Debug Commands
```bash
# Clear cache
rm -rf cache/*

# Reset environment
poetry env remove python
poetry install --with dev

# Verbose testing
pytest -v -s
```

## 📊 CI/CD Pipeline

The project uses GitHub Actions for continuous integration:

- **Pull Request**: Lint, test, security checks
- **Main Branch**: Full test suite, build, deploy docs
- **Tags**: Create releases, build Docker images

### Workflow Files
- `.github/workflows/ci.yml` - Main CI/CD pipeline
- `.github/workflows/release.yml` - Release automation
- `.github/workflows/codeql.yml` - Security analysis

## 📚 Additional Resources

- **Poetry Documentation**: https://python-poetry.org/docs/
- **pytest Documentation**: https://docs.pytest.org/
- **Black Documentation**: https://black.readthedocs.io/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
