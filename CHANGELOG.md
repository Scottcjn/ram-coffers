# Changelog

All notable changes to RAM Coffers will be documented in this file.

## [1.0.0] - 2026-02-06

### Added
- Initial release of RAM Coffers
- NUMA-distributed weight banking architecture
- Resonance routing for domain-based coffer selection
- Non-bijunctive pruning (PSE collapse)
- DCBT resident prefetch hints for POWER8
- GGUF model sharding support

### Performance
- 8.81x speedup over stock llama.cpp on POWER8
- 147.54 tokens/sec (pp128) on TinyLlama 1.1B Q4

### Technical Details
- Compatible with IBM POWER8 S824
- Uses vec_perm AltiVec instructions
- NUMA-aware memory allocation
- Supports model sizes up to available RAM

### Related Publications
- RAM Coffers: NUMA-Distributed Weight Banking (DOI: 10.5281/zenodo.18321905)
- Non-Bijunctive Permutation Collapse (DOI: 10.5281/zenodo.18623920)
- PSE Hardware Entropy (DOI: 10.5281/zenodo.18623922)
