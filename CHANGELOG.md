# Changelog

All notable changes to QFrame will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Grid Trading Strategy implementation
- Freqtrade integration backend
- WebUI Dashboard for monitoring
- Real-time data streaming pipeline

## [0.1.0] - 2024-09-22

### Added
- 🏗️ **Core Framework Architecture**
  - Dependency Injection Container with thread safety
  - Pydantic v2 configuration system with environment support
  - Protocol-based interfaces for clean contracts
  - Comprehensive test suite (73 tests passing)

- 🧠 **Research Strategies**
  - DMN LSTM Strategy with PyTorch implementation
  - Adaptive Mean Reversion with regime detection
  - Funding Arbitrage with ML prediction
  - RL Alpha Generator based on academic research

- 📊 **Feature Engineering**
  - 15+ symbolic operators from research papers
  - Alpha formula implementations (Alpha006, Alpha061, Alpha099)
  - Robust handling of NaN data and edge cases
  - Time series and cross-sectional operators

- 🔧 **Development Infrastructure**
  - Poetry dependency management
  - Type-safe configuration with Pydantic
  - Comprehensive testing with pytest
  - Code quality tools (Black, Ruff, MyPy)

- 📚 **Documentation**
  - Complete setup guides (Poetry, TA-Lib)
  - Architectural documentation in CLAUDE.md
  - Contributing guidelines
  - Security policy

### Technical Details
- Python 3.11+ support
- Modern async/await patterns
- Thread-safe operations
- Memory-efficient data processing
- Extensible plugin architecture

### Dependencies
- Core: pandas, numpy, pydantic, typer
- ML: torch, scikit-learn, scipy
- Trading: ta-lib, ccxt
- Testing: pytest, hypothesis
- Quality: black, ruff, mypy

## [0.0.1] - Initial Setup

### Added
- Initial repository structure
- Basic project configuration

---

**Legend:**
- 🏗️ Architecture & Infrastructure
- 🧠 Trading Strategies & Research
- 📊 Data & Features
- 🔧 Development Tools
- 📚 Documentation
- 🐛 Bug Fixes
- ⚡ Performance Improvements
- 🔒 Security Updates