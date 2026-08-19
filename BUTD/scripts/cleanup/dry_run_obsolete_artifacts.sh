#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}" || exit 1

STAMP=${STAMP:-$(date +%Y%m%d)}
ARCHIVE_DIR="archive/obsolete_artifacts/${STAMP}"
README_PATH="${ARCHIVE_DIR}/README.md"

mkdir -p "${ARCHIVE_DIR}"

cat > "${README_PATH}" <<EOF
# Obsolete Artifact Dry Run ${STAMP}

This is a dry-run manifest only. No files are moved or deleted by this script.

Review the candidates below before creating any archival or removal command.
EOF

{
  printf "\n## Candidate legacy ACD/DHC/S2S scripts\n\n"
  find scripts -type f \( -name '*acd*' -o -name '*dhc*' -o -name '*s2s*' \) \
    ! -path 'scripts/new_method_v2/*' -print 2>/dev/null || true
  printf "\n## Candidate legacy docs\n\n"
  find docs "new plan" "new plan md" -type f \( -iname '*acd*' -o -iname '*dhc*' -o -iname '*s2s*' \) \
    -print 2>/dev/null || true
  printf "\n## Candidate root legacy entrypoints\n\n"
  find . -maxdepth 1 -type f \( -name '*acd*' -o -name '*dhc*' -o -name '*s2s*' -o -name 'run_priority_experiments.sh' \) \
    -print 2>/dev/null || true
  printf "\n## Candidate pycache files\n\n"
  find . -path './.git' -prune -o -path './archive' -prune -o -type d -name __pycache__ -print
  printf "\n## Candidate old logs\n\n"
  find logs -maxdepth 4 -type f -name '*.log' -print 2>/dev/null || true
} | tee -a "${README_PATH}"

printf "Dry-run manifest written to %s\n" "${README_PATH}"
