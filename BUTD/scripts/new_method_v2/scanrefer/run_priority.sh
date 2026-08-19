#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/two_stage/run_priority.sh" "$@"
bash "${SCRIPT_DIR}/single_stage/run_priority.sh" "$@"
