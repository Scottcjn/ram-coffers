# Contributing to RAM Coffers

Thank you for your interest in contributing to RAM Coffers! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Architecture Overview](#architecture-overview)

## Code of Conduct

This project follows standard open-source community guidelines. Be respectful, constructive, and inclusive in all interactions.

## How Can I Contribute?

### Reporting Bugs

- Check existing issues before creating a new one
- Use the bug report template if available
- Include system details: OS, NUMA topology, RAM configuration
- Provide steps to reproduce

### Suggesting Enhancements

- Check if the enhancement has already been suggested
- Clearly describe the feature and its benefits
- Explain how it fits with the NUMA-distributed architecture

### Pull Requests

- Fork the repository
- Create a feature branch
- Make your changes
- Test on NUMA hardware if possible
- Submit a pull request with clear description

## Development Setup

### Prerequisites

- IBM POWER8 system or NUMA-compatible hardware
- Python 3.8+
- NumPy, SciPy for numerical operations
- NUMA development libraries

### Setup Steps

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ram-coffers.git
   cd ram-coffers
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Check NUMA topology:
   ```bash
   numactl --hardware
   ```

4. Run benchmarks:
   ```bash
   python benchmark.py
   ```

## Pull Request Process

1. **Branch naming**: Use descriptive names like `feature/add-coffer-metrics` or `fix/resonance-routing`
2. **Commit messages**: Follow conventional commits format
3. **Testing**: Include tests for new functionality
4. **Documentation**: Update README and docs as needed
5. **Review**: Address all review comments before merge

## Architecture Overview

RAM Coffers uses NUMA-distributed memory banking:

```
| Coffer | NUMA Node | Capacity | Role                |
|--------|-----------|----------|---------------------|
| 0      | 3         | 193 GB   | Heavy/General (core)|
| 1      | 1         | 183 GB   | Science/Tech domain |
| 2      | 0         | 119 GB   | Creative/Long CTX   |
| 3      | 2         | 62 GB    | Niche/History       |
```

Key concepts:
- **Resonance Routing**: Query-to-coffer matching via cosine similarity
- **DCBT Prefetch**: PowerPC cache hints for L2/L3 residency
- **PSE Entropy**: Hardware entropy injection for inference

## License

By contributing, you agree that your contributions will be licensed under the project's license.

---

📚 For more details, see the [publication list](https://doi.org/10.5281/zenodo.18321905)
