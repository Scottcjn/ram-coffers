# Contributing to RAM Coffers

Thank you for your interest in contributing to RAM Coffers! This document provides guidelines and instructions for contributing to this NUMA-distributed conditional memory architecture for LLM inference.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Style Guidelines](#style-guidelines)
- [Questions and Support](#questions-and-support)

## Code of Conduct

This project adheres to the [Contributor Covenant 3.0 Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to uphold this code. Please report unacceptable behavior to the project maintainers.

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Use the issue templates in `.github/ISSUE_TEMPLATE/`
- Provide clear, descriptive titles
- Include steps to reproduce bugs
- Mention your hardware setup (NUMA configuration, CPU architecture)

### Suggesting Enhancements

- Open an issue with the "enhancement" label
- Describe the feature and its benefits
- Explain how it fits with RAM Coffers' architecture
- Reference relevant research if applicable

### Contributing Code

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the style guidelines below
3. **Test your changes** on appropriate hardware if possible
4. **Submit a pull request** with a clear description

### Areas We Need Help With

- **Portability**: Testing on different NUMA architectures (AMD EPYC, Intel Xeon)
- **Documentation**: Improving code comments and examples
- **Optimization**: Additional SIMD implementations (AVX-512, NEON)
- **Testing**: Benchmarking and validation frameworks
- **Integration**: Examples with different LLM frameworks

## Development Setup

### Prerequisites

- **C Compiler**: GCC or Clang with C11 support
- **NUMA Libraries**: `libnuma` (Linux)
- **Hardware**: NUMA-capable system recommended for testing
- **Git**: For version control

### Getting Started

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/ram-coffers.git
cd ram-coffers

# Add upstream remote
git remote add upstream https://github.com/Scottcjn/ram-coffers.git

# Keep your fork updated
git fetch upstream
git checkout main
git merge upstream/main
```

### Project Structure

```
ram-coffers/
├── ggml-ram-coffers.h        # Multi-bank NUMA routing
├── ggml-coffer-mmap.h        # Memory mapping and sharding
├── ggml-ram-coffer.h         # Single coffer implementation
├── ggml-intelligent-collapse.h # Hebbian path collapse
├── ggml-topk-collapse-vsx.h  # VSX-optimized collapse
├── pse-entropy-burst.h       # Hardware entropy injection
├── power8-compat.h           # POWER9→POWER8 compatibility
└── grail-v-paper/            # GRAIL-V research paper
```

### Code Reading Path

For new contributors, we recommend this order:

1. `ggml-ram-coffers.h` — High-level routing and coffer selection
2. `ggml-coffer-mmap.h` — Memory mapping and NUMA placement
3. `ggml-topk-collapse-vsx.h` — Vectorized collapse details
4. `power8-compat.h` — ISA compatibility layer

### Testing Your Changes

Since this is a header-only library focused on NUMA architectures:

1. **Compile test**: Ensure headers compile without errors
2. **Integration test**: Test with a llama.cpp build if possible
3. **NUMA test**: Verify behavior on NUMA hardware
4. **Performance test**: Benchmark changes against baseline

## Pull Request Guidelines

### Before Submitting

- [ ] Code compiles without warnings
- [ ] Changes are focused and atomic
- [ ] Commit messages are clear and descriptive
- [ ] Documentation is updated if needed
- [ ] No unrelated changes included

### PR Title Format

Use conventional commit prefixes:

- `feat:` — New features
- `fix:` — Bug fixes
- `docs:` — Documentation changes
- `perf:` — Performance improvements
- `refactor:` — Code refactoring
- `test:` — Testing improvements

Examples:
- `feat: Add AVX-512 collapse implementation`
- `fix: Correct NUMA node affinity in mmap`
- `docs: Clarify resonance routing algorithm`

### PR Description Template

```markdown
## Summary
Brief description of changes

## Motivation
Why this change is needed

## Changes
- List of specific changes

## Testing
How you tested these changes

## Hardware
What hardware you tested on (if applicable)
```

### Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged

## Style Guidelines

### C Code Style

- **Indentation**: 4 spaces (no tabs)
- **Line length**: Keep lines under 100 characters
- **Comments**: Use `/* */` for block comments, `//` for inline
- **Naming**: 
  - `snake_case` for functions and variables
  - `UPPER_CASE` for macros and constants
  - Descriptive names preferred over short ones

### Example

```c
/* Calculate cosine similarity between two embedding vectors */
static float calc_resonance_score(
    const float *query_embedding,
    const float *coffer_signature,
    int embedding_dim
) {
    float dot_product = 0.0f;
    float query_norm = 0.0f;
    float sig_norm = 0.0f;
    
    for (int i = 0; i < embedding_dim; i++) {
        dot_product += query_embedding[i] * coffer_signature[i];
        query_norm += query_embedding[i] * query_embedding[i];
        sig_norm += coffer_signature[i] * coffer_signature[i];
    }
    
    return dot_product / (sqrtf(query_norm) * sqrtf(sig_norm));
}
```

### Documentation Style

- Keep README.md concise; detailed docs go in separate files
- Include code examples for new features
- Reference relevant research papers when applicable
- Update the file list in README.md for new headers

## Questions and Support

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Contact**: 
  - GitHub: [@Scottcjn](https://github.com/Scottcjn)
  - X/Twitter: @RustchainPOA

---

Thank you for helping make RAM Coffers better! Your contributions help advance efficient LLM inference on diverse hardware architectures.
