#!/usr/bin/env bash

# Run from anywhere: anchor to this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
#
# build_matrix.sh - Verify RAM Coffers builds and runs across capability modes.
#
# Run from anywhere:  bash tests/build_matrix.sh   (or ./build_matrix.sh from tests/)
#
# Modes exercised:
#   1. auto-detect                (whatever this machine has)
#   2. NUMA forced OFF            (-DGGML_COFFERS_NO_NUMA, no -lnuma)
#   3. NUMA macro forced to 0     (-DGGML_COFFERS_HAVE_NUMA=0)
#   4. AltiVec forced OFF         (-DGGML_COFFERS_NO_ALTIVEC)
#   5. Everything forced OFF      (uniform + scalar)
#   6. C++ compilation            (headers must be usable from C++ too)
#
set -u

CC="${CC:-gcc}"
CXX="${CXX:-g++}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Quiet the noise that is expected in a header-only library.
QUIET_WARNS="-Wno-unused-function -Wno-unused-parameter -Wno-unknown-pragmas"

pass=0
fail=0

have_numa_lib() {
    echo 'int main(void){return 0;}' > "$TMP/n.c"
    $CC "$TMP/n.c" -o "$TMP/n" -lnuma 2>/dev/null
}

run_case() {
    local name="$1"; shift
    local src="$1";  shift
    local out="$TMP/bin_$$"

    printf '\n=== %s ===\n' "$name"
    printf 'cc: %s %s\n' "$CC" "$*"

    if ! $CC -std=c11 -I.. -Wall -Wextra $QUIET_WARNS "$src" -o "$out" "$@" 2>"$TMP/err"; then
        echo "BUILD FAILED:"; cat "$TMP/err"; fail=$((fail+1)); return
    fi
    if [ -s "$TMP/err" ]; then echo "warnings:"; cat "$TMP/err"; fi

    if "$out" > "$TMP/out" 2>&1; then
        grep -E 'ALL TESTS PASSED|FAILURES PRESENT|uniform-memory mode|NUMA nodes' "$TMP/out" || true
        echo "RESULT: PASS"; pass=$((pass+1))
    else
        echo "RUN FAILED:"; tail -30 "$TMP/out"; fail=$((fail+1))
    fi
}

run_cxx_case() {
    local name="$1"; shift
    local src="$1";  shift
    printf '\n=== %s ===\n' "$name"
    cp "$src" "$TMP/cxx.cpp"
    if $CXX -std=c++17 -I.. -Wall $QUIET_WARNS "$TMP/cxx.cpp" -o "$TMP/cxxbin" "$@" 2>"$TMP/cxxerr"; then
        echo "RESULT: PASS (compiles as C++)"; pass=$((pass+1))
    else
        echo "BUILD FAILED:"; head -30 "$TMP/cxxerr"; fail=$((fail+1))
    fi
}

NUMA_LDFLAG=""
if have_numa_lib; then
    NUMA_LDFLAG="-lnuma"
    echo "libnuma: PRESENT"
else
    echo "libnuma: ABSENT (auto-detect cases will use the uniform path)"
fi

for SRC in test_portability.c test_coffer_headers.c; do
    echo
    echo "##############################################"
    echo "# $SRC"
    echo "##############################################"

    run_case "1. auto-detect"            "$SRC" -lm $NUMA_LDFLAG
    run_case "2. NUMA forced off"        "$SRC" -DGGML_COFFERS_NO_NUMA -lm
    run_case "3. HAVE_NUMA=0"            "$SRC" -DGGML_COFFERS_HAVE_NUMA=0 -lm
    run_case "4. AltiVec forced off"     "$SRC" -DGGML_COFFERS_NO_ALTIVEC -lm $NUMA_LDFLAG
    run_case "5. all fallbacks forced"   "$SRC" -DGGML_COFFERS_NO_NUMA -DGGML_COFFERS_NO_ALTIVEC -lm
done

run_cxx_case "6. C++ (test_portability)" test_portability.c -lm $NUMA_LDFLAG

printf '\n==============================\n'
printf 'MATRIX: %d passed, %d failed\n' "$pass" "$fail"
printf '==============================\n'
[ "$fail" -eq 0 ]
