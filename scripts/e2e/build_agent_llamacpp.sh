#!/usr/bin/env bash
# Moved to e2e/adapters/llamacpp/build_agent.sh; this runs it with the same arguments.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../e2e/adapters/llamacpp/build_agent.sh" "$@"
