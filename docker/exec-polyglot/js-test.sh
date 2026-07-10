#!/usr/bin/env bash
# Run an Exercism JS exercise's tests offline using the globally-installed jest/babel (NODE_PATH),
# avoiding the per-exercise `npm install` that --network=none would block. Invoked by the plugin as
# the javascript test command; runs in /work (the mounted exercise dir).
set -o pipefail
cd /work || exit 2
# Exercism JS exercises ship *.spec.js test files and a babel.config.js; the global jest picks them
# up. --ci for stable output; --rootDir . to scope collection to this exercise.
exec jest --ci --rootDir . 2>&1
