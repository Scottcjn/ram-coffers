# RAM Coffers: NUMA-Distributed Conditional Memory for LLM Inference

[![BCOS Certified](https://img.shields.io/badge/BCOS-Certified-brightgreen?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMiAxTDMgNXY2YzAgNS41NSAzLjg0IDEwLjc0IDkgMTIgNS4xNi0xLjI2IDktNi40NSA5LTEyVjVsLTktNHptLTIgMTZsLTQtNCA1LjQxLTUuNDEgMS40MSAxLjQxTDEwIDE0bDYtNiAxLjQxIDEuNDFMMTAgMTd6Ii8+PC9zdmc+)](BCOS.md)
**Author:** Scott Boudreaux
**Date:** December 16, 2025
**Institution:** Elyan Labs (Independent Research)
**Hardware:** IBM POWER8 S824 (320GB RAM, Dual 8-core)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18321905.svg)](https://doi.org/10.5281/zenodo.18321905)

## Overview

RAM Coffers implements a NUMA-aware memory architecture that conditionally distributes LLM weights across memory domains based on access patterns and inference requirements. This system optimizes memory bandwidth utilization by strategically placing frequently accessed model parameters closer to processing cores while maintaining coherent weight synchronization across NUMA nodes.

## Publications

| Paper | DOI | Date |
|-------|-----|------|
| **RAM Coffers: NUMA-Distributed Weight Banking** | [10.5281/zenodo.18321905](https://doi.org/10.5281/zenodo.18321905) | Jan 2026 |
| **Non-Bijunctive Permutation Collapse** (vec_perm for LLM attention) | [10.5281/zenodo.18623920](https://doi.org/10.5281/zenodo.18623920) | Feb 2026 |
| **PSE Hardware Entropy for Behavioral Divergence** (mftb injection) | [10.5281/zenodo.18623922](https://doi.org/10.5281/zenodo.18623922) | Feb 2026 |
| **Neuromorphic Prompt Translation** (GRAIL-V, emotional prompting) | [10.5281/zenodo.18623594](https://doi.org/10.5281/zenodo.18623594) | Feb 2026 |
| **RustChain: One CPU, One Vote** (Proof of Antiquity consensus) | [10.5281/zenodo.18623592]