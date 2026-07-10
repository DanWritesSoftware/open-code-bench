#!/usr/bin/env bash
# Build and run an Exercism C++ exercise's tests offline. Exercism cpp exercises ship a CMakeLists
# (with a bundled Catch2 header) and gate the full test set behind the EXERCISM_RUN_ALL_TESTS macro.
# Invoked by the plugin as the cpp test command; runs in /work (the mounted exercise dir).
set -o pipefail
cd /work || exit 2
cmake -B build -DEXERCISM_RUN_ALL_TESTS=1 . 2>&1 || { echo "cmake configure failed"; exit 1; }
cmake --build build 2>&1 || { echo "build failed"; exit 1; }
# Prefer ctest (Exercism CMakeLists usually enable_testing()); fall back to any built test binary.
if ctest --test-dir build --output-on-failure 2>&1; then
    exit 0
fi
bin="$(find build -maxdepth 2 -type f -perm -u+x ! -name '*.so' 2>/dev/null | head -n1)"
if [ -n "$bin" ]; then
    "$bin"; exit $?
fi
exit 1
