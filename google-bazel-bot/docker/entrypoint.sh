#!/usr/bin/env bash

set -eu
set -o pipefail

USER=buildkite-agent
P="${BUILDKITE_BUILD_PATH:=/var/lib/buildkite-agent}"
mkdir -p "$P"
chown -R ${USER}:${USER} "$P"

# Run with tini to correctly pass exit codes.
exec /usr/bin/tini -g -- "$@"
