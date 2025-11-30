#!/bin/bash
# Run SatStash TUI (wrapper around sxm_app)

cd "$(dirname "$0")"

exec ./sxm_app "$@"
