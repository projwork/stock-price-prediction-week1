# CI/CD Setup and Troubleshooting Guide

## Overview

This document provides comprehensive guidance for setting up CI/CD for the Financial Analysis project, including solutions for common dependency issues.

## CI/CD Architecture

### Files Structure

```
.github/
└── workflows/
    └── ci.yml                 # Main CI pipeline
requirements.txt               # Development requirements
requirements-ci.txt           # CI-optimized requirements
src/
├── technical_analyzer.py     # Full TA-Lib analyzer
└── simple_technical_analyzer.py  # Fallback analyzer
```

## Common Issues and Solutions

### 1. Python Version Compatibility

**Problem**:

```
ERROR: Ignored the following versions that require a different python version: 1.21.2 Requires-Python >=3.7,<3.11
```

**Solution**:

- Updated CI to test Python 3.9, 3.10, 3.11 (avoiding 3.12 compatibility issues)
- Created version ranges in requirements that work across these versions
- Used conservative version constraints in `requirements-ci.txt`

### 2. Windows-Specific Dependencies

**Problem**:

```
ERROR: Could not find a version that satisfies the requirement pywin32==310
```

**Solution**:

```python
# In requirements.txt
pywin32>=305; platform_system == "Windows"
```

This ensures the package only installs on Windows systems.

### 3. TA-Lib Installation Issues

**Problem**: TA-Lib requires system-level C libraries not available in CI.

**Solutions**:

#### Option A: Install System Dependencies (Recommended)

```yaml
# Ubuntu
- name: Install system dependencies (Ubuntu)
  if: matrix.os == 'ubuntu-latest'
  run: |
    sudo apt-get update
    sudo apt-get install -y build-essential wget
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install

# macOS
- name: Install system dependencies (macOS)
  if: matrix.os == 'macos-latest'
  run: brew install ta-lib
```

#### Option B: Use Fallback Analyzer

If TA-Lib installation fails, the project includes `SimpleTechnicalAnalyzer` that provides the same functionality using only pandas/numpy.

## Requirements Files

### requirements.txt (Development)

Full-featured requirements for local development with flexible version ranges.

### requirements-ci.txt (CI/CD)

Conservative version constraints optimized for CI environments:

```python
# Core libraries with proven CI compatibility
pandas>=2.0.0,<2.2.0
numpy>=1.24.0,<1.26.0
scipy>=1.10.0,<1.12.0
```

## CI Pipeline Features

### Multi-Platform Testing

- Ubuntu (Linux)
- Windows
- macOS

### Multi-Python Version Testing

- Python 3.9
- Python 3.10
- Python 3.11

### Intelligent Dependency Handling

```yaml
- name: Install Python dependencies (CI version)
  run: |
    if [ -f requirements-ci.txt ]; then
      pip install -r requirements-ci.txt
    else
      pip install -r requirements.txt
    fi

- name: Install TA-Lib (with fallback)
  run: |
    pip install TA-Lib || echo "TA-Lib installation failed, continuing without it"
```

### Comprehensive Testing

1. **Import Tests**: Verify all critical packages import correctly
2. **Functionality Tests**: Basic operations work as expected
3. **Linting**: Code quality checks with flake8, black, isort

## Local Development Setup

### Option 1: Full Installation

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Install Python packages
pip install -r requirements.txt
```

### Option 2: Minimal Installation (No TA-Lib)

```bash
# Install CI requirements (works without TA-Lib)
pip install -r requirements-ci.txt

# Use SimpleTechnicalAnalyzer instead of TechnicalAnalyzer
from src.simple_technical_analyzer import SimpleTechnicalAnalyzer
```

## Troubleshooting Common CI Failures

### 1. Package Version Conflicts

**Symptom**: `ResolutionImpossible` errors during pip install

**Solution**:

1. Use `requirements-ci.txt` with conservative versions
2. Add version ranges instead of exact pins
3. Check Python version compatibility

### 2. Missing System Dependencies

**Symptom**: Compilation errors for native packages

**Solution**:

1. Add system package installation steps to CI
2. Use pre-compiled wheels when available
3. Implement fallback solutions

### 3. Platform-Specific Issues

**Symptom**: Package works on one OS but not others

**Solution**:

```python
# Use platform markers in requirements
package>=1.0.0; platform_system == "Linux"
different-package>=2.0.0; platform_system == "Windows"
```

### 4. Memory/Timeout Issues

**Symptom**: CI jobs timeout or run out of memory

**Solution**:

1. Use package caching
2. Minimize test data size
3. Split large jobs into smaller ones

## Best Practices

### 1. Version Management

- Use version ranges, not exact pins
- Test across multiple Python versions
- Keep CI requirements conservative

### 2. Platform Independence

- Use platform markers for OS-specific packages
- Test on multiple operating systems
- Provide fallback implementations

### 3. Dependency Management

- Separate development and CI requirements
- Document system dependencies
- Implement graceful degradation

### 4. CI Optimization

- Use caching for package installations
- Cache pip packages and system dependencies
- Parallel job execution where possible

## Integration with Your Project

### Using the Technical Analyzers

```python
# Try full-featured analyzer first
try:
    from src.technical_analyzer import TechnicalAnalyzer
    analyzer = TechnicalAnalyzer(data)
    print("Using full TA-Lib analyzer")
except ImportError:
    from src.simple_technical_analyzer import SimpleTechnicalAnalyzer
    analyzer = SimpleTechnicalAnalyzer(data)
    print("Using simple fallback analyzer")

# Both provide similar interfaces
analysis = analyzer.get_comprehensive_analysis()
signals = analyzer.generate_simple_signals()  # Only in simple analyzer
```

### Environment Detection

```python
import sys
import os

def get_environment_info():
    """Get information about the current environment."""
    return {
        'python_version': sys.version,
        'platform': sys.platform,
        'ci_environment': 'CI' in os.environ,
        'github_actions': 'GITHUB_ACTIONS' in os.environ
    }
```

## Monitoring and Maintenance

### Regular Tasks

1. **Update Dependencies**: Review and update version ranges quarterly
2. **Test New Python Versions**: Add new Python versions as they're released
3. **Monitor CI Performance**: Track build times and failure rates
4. **Security Updates**: Keep dependencies updated for security patches

### Health Checks

- Monitor CI success rates
- Check for new platform compatibility issues
- Validate cross-platform functionality
- Review dependency security advisories

## Support and Resources

### Documentation

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pip Environment Markers](https://peps.python.org/pep-0508/#environment-markers)
- [TA-Lib Installation Guide](https://github.com/mrjbq7/ta-lib#installation)

### Common Commands

```bash
# Test CI locally with act
act -P ubuntu-latest=nektos/act-environments-ubuntu:18.04

# Check package compatibility
pip-check

# Validate requirements syntax
pip install -r requirements-ci.txt --dry-run

# Generate pip freeze for exact versions
pip freeze > requirements-frozen.txt
```

This setup ensures reliable CI/CD operations while maintaining full functionality for development environments.
