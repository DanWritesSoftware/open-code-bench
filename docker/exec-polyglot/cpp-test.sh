#!/usr/bin/env bash
# Build and run an Exercism C++ exercise's tests offline. Exercism cpp exercises ship a CMakeLists
# (with a bundled Catch2 header) and gate the full test set behind the EXERCISM_RUN_ALL_TESTS macro.
# Invoked by the plugin as the cpp test command; runs in /work (the mounted exercise dir).
set -o pipefail
cd /work || exit 2
# Exercism's CMakeLists derives the exercise name from the source dir NAME (get_filename_component
# ... NAME), then `string(REPLACE "-" "_" file ${exercise})` and looks for ${file}_test.cpp. We
# bind-mount the exercise at /work, so the name would wrongly become "work". Recreate a dir whose
# LEAF name is exactly the slug (taken from the *_test.cpp file, which is already underscore-form)
# and build there. The leaf must be the bare slug with NO prefix: any prefix survives REPLACE and
# makes CMake look for <prefix>_<slug>_test.cpp, which doesn't exist.
testfile="$(ls *_test.cpp 2>/dev/null | head -n1)"
slug="${testfile%_test.cpp}"
if [ -n "$slug" ] && [ "$slug" != "work" ]; then
    dest="/tmp/cppbuild/$slug"
    rm -rf "$dest" && mkdir -p "$dest" && cp -a /work/. "$dest/" && cd "$dest" || exit 2
fi
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
