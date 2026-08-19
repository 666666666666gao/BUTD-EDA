#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/sr3d/run_priority.sh" "$@"
bash "${SCRIPT_DIR}/nr3d/run_priority.sh" "$@"
bash "${SCRIPT_DIR}/scanrefer/run_priority.sh" "$@"
