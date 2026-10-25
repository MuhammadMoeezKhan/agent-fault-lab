FROM node:22-bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates python3 \
      python3-venv curl make g++ \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://astral.sh/uv/0.5.11/install.sh | \
      env UV_INSTALL_DIR=/usr/local/bin sh

RUN useradd --create-home --home-dir /home/afl --uid 10001 afl
RUN mkdir -p /work && chown afl:afl /work

ENV npm_config_update_notifier=false

# Install scripts must run: several servers depend on native modules
# (better-sqlite3) that only exist once node-gyp has built them, and skipping
# the build produces a server that starts and then fails on its first call,
# which is worse than one that does not start. Optional dependencies are
# omitted because some servers pull fsevents, a macOS-only module that cannot
# build on Linux and only accelerates file watching where it does exist.
#
# A package that still fails is recorded rather than fatal, because one
# unbuildable server should not cost the image every other server. The audit
# refuses to run a server whose install failed, so a missing one is loud at
# run time instead of silent.
ARG PACKAGES=""
RUN set -u; mkdir -p /opt/afl; : > /opt/afl/installed; : > /opt/afl/failed; \
    for pkg in $PACKAGES; do \
      echo "installing $pkg"; \
      if npm install -g --omit=optional "$pkg" >/tmp/npm.log 2>&1; then \
        echo "$pkg" >> /opt/afl/installed; \
      else \
        echo "$pkg" >> /opt/afl/failed; \
        echo "FAILED $pkg:"; tail -5 /tmp/npm.log; \
      fi; \
    done; \
    echo "--- installed ---"; cat /opt/afl/installed; \
    echo "--- failed ---"; cat /opt/afl/failed

# Python servers are installed the same way and recorded in the same files.
# `uvx mcp-server-git` cannot run inside the container: uvx resolves the
# package against the index on every launch, which fails with the network
# off at run time, exactly as `npx` does. `uv tool install` with the bin
# directory on PATH leaves a plain executable behind, which works whatever
# HOME is set to at run time.
ARG PY_PACKAGES=""
RUN set -u; for pkg in $PY_PACKAGES; do \
      echo "installing $pkg"; \
      if UV_TOOL_BIN_DIR=/usr/local/bin uv tool install "$pkg" \
            >/tmp/uv.log 2>&1; then \
        echo "$pkg" >> /opt/afl/installed; \
      else \
        echo "$pkg" >> /opt/afl/failed; \
        echo "FAILED $pkg:"; tail -5 /tmp/uv.log; \
      fi; \
    done; \
    echo "--- installed ---"; cat /opt/afl/installed; \
    echo "--- failed ---"; cat /opt/afl/failed

USER afl
ENV HOME=/home/afl
WORKDIR /work
