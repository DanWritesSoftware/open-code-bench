# open-code-bench — hardened, OFFLINE multi-language exec image for the Aider polyglot benchmark (D11).
#
# Unlike the other three exec images (one language each), polyglot exercises span six toolchains,
# so this image carries all of them: Python, Go, Rust, Java (JDK+Gradle), Node/JS (Jest), C++
# (g++/CMake/Catch2). The runner ships one exercise dir at a time to /work and runs that language's
# test command inside it (see AiderPolyglot.TEST_COMMANDS); the container runs with --network=none,
# so every test-time dependency must already be present in the image.
#
# The dependency-warming problem, per language:
#   * python / go / rust / cpp — Exercism practice exercises are (almost all) stdlib-only, so their
#     tests run offline against the bare toolchain. Rust/Go are invoked with their offline flags.
#   * javascript — every JS exercise shares the same devDeps (jest + babel). We install those ONCE
#     globally and point js-test.sh at the global jest, so no per-exercise `npm install` is needed.
#   * java — exercises use the Gradle wrapper + JUnit5. We install Gradle and warm ~/.gradle with a
#     throwaway build that pulls the JUnit BOM, then run system `gradle test --offline` at runtime
#     (system gradle == the wrapper's pinned 8.7, so we skip the wrapper's network bootstrap).
#
# Build (on the amd64 sandbox host, NOT the arm64 Spark), network ON to warm caches:
#   docker build -t ocb-polyglot-exec:0.1.0 docker/exec-polyglot
#
# NOTE (parity with the other exec images at first commit): this image is NOT built/validated yet.
# Expect a debug cycle when first built on the sandbox host — the JS (jest/babel resolution under
# NODE_PATH) and Java (gradle offline cache) paths are the most likely to need iteration.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git xz-utils unzip procps coreutils \
        python3 python3-pip python3-pytest \
        golang-go \
        g++ cmake make \
        openjdk-17-jdk-headless \
        nodejs npm \
        cargo \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ---- JavaScript: one global jest/babel install shared by every JS exercise ----
# Exercism JS practice exercises all declare the same devDependencies; installing them globally and
# exposing them via NODE_PATH lets js-test.sh run jest offline with no per-exercise npm install.
ENV NODE_PATH=/usr/local/lib/node_modules
# Exercism JS exercises' babel.config.js references '@exercism/babel-preset-javascript' (not the
# bare @babel/preset-env); install that preset (it pulls @babel/preset-env + core-js as deps) so
# babel can resolve it under NODE_PATH offline.
RUN npm install -g --no-audit --no-fund \
        jest@29 babel-jest@29 \
        @babel/core@7 @babel/preset-env@7 \
        @exercism/babel-preset-javascript@0.2 core-js@3 \
    && npm cache verify || true

# ---- Java: install Gradle + warm the JUnit cache offline-ready ----
ARG GRADLE_VERSION=8.7
RUN curl -fsSL "https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip" -o /tmp/g.zip \
    && mkdir -p /opt/gradle && (cd /opt/gradle && unzip -q /tmp/g.zip) && rm /tmp/g.zip
ENV PATH=/opt/gradle/gradle-${GRADLE_VERSION}/bin:$PATH
# The runner invokes test commands via `bash -lc`, whose login shell resets PATH from /etc/profile
# and drops the ENV PATH above — so symlink gradle where the login PATH always looks (mirrors the
# python3 symlink), letting `gradle test` resolve at runtime.
RUN ln -sf /opt/gradle/gradle-${GRADLE_VERSION}/bin/gradle /usr/local/bin/gradle
# Warm ~/.gradle by resolving JUnit5 once (network on at build); runtime uses --offline.
COPY warm-gradle /opt/ocb-warm-gradle
# Run a real `gradle test` (not just `dependencies`) so compileTestJava + test execution download
# EVERY runtime jar (platform launcher/engine, jupiter api/engine, assertj) into the cache — the
# metadata-only `dependencies` task leaves those out and breaks the offline runtime test.
RUN cd /opt/ocb-warm-gradle && gradle --no-daemon test >/dev/null 2>&1 || true

# ---- Rust / Go offline defaults ----
ENV CARGO_NET_OFFLINE=true \
    GOFLAGS=-mod=mod \
    GOTOOLCHAIN=local \
    PYTHONDONTWRITEBYTECODE=1

# helper test scripts the plugin invokes for js / cpp
COPY js-test.sh cpp-test.sh /opt/ocb/
RUN chmod +x /opt/ocb/js-test.sh /opt/ocb/cpp-test.sh

# Non-root (D11). Copy the warmed caches into the runtime user's HOME so --network=none tests find them.
RUN useradd -m poly \
    && cp -r /root/.gradle /home/poly/.gradle 2>/dev/null || true \
    && mkdir -p /home/poly/.cargo /home/poly/go \
    && chown -R poly:poly /home/poly
USER poly
ENV HOME=/home/poly GRADLE_USER_HOME=/home/poly/.gradle
WORKDIR /work
# No ENTRYPOINT: the runner passes a full `bash -lc "cd /work && <test cmd>"` as the command.
