## Quick Start

This section provides a simple example for getting started with RAM Coffers.

### Building

```bash
# Clone the repository
git clone https://github.com/Scottcjn/ram-coffers.git
cd ram-coffers

# Build with CMake
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Running

```bash
# Run with a model
./ram-coffers --model <path-to-model> --numa-optimized
```

### Verification

Check that NUMA optimization is active:

```bash
# Should show coffer distribution
./ram-coffers --info
```

---

*Added 2026-02-20 for improved documentation clarity.*
