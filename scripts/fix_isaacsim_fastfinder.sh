#!/usr/bin/env bash
# Workaround for a Kit (Isaac Sim 6.2) FastFinder issue.
#
# Kit's FastFinder (omni/ext/_impl/fast_importer.py) registers whichever file it meets first
# while scanning extension folders in unsorted os.scandir() order. Standard Python always
# prefers a compiled module (X.cpython-312-*.so) over X.py, but FastFinder may pick X.py.
# When a package ships both, compiled and pure-Python modules get mixed, e.g.
#   ValueError: charset_normalizer.md.MessDetectorPlugin size changed ...
# which makes ~40 Isaac Sim extensions fail to load. Whether it happens depends on the
# filesystem's directory order, so it shows up on some machines and not on others.
#
# This script renames every X.py that has a sibling X.cpython-312-*.so under the Isaac Sim
# package to X.py.disabled, so Kit loads the same file standard Python would. It only renames
# files (the uv cache is untouched) and is safe to run repeatedly.
#
# Usage (from the project root, after `uv sync --extra isaacsim`):
#   bash scripts/fix_isaacsim_fastfinder.sh            # apply
#   bash scripts/fix_isaacsim_fastfinder.sh --revert   # undo
# Run it again whenever `uv sync` reinstalls Isaac Sim packages.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACSIM_DIR="${ISAACSIM_DIR:-$PROJECT_ROOT/.venv/lib/python3.12/site-packages/isaacsim}"
MODE="${1:-apply}"

if [[ ! -d "$ISAACSIM_DIR" ]]; then
  echo "Isaac Sim not found at: $ISAACSIM_DIR" >&2
  echo "Run 'uv sync --extra isaacsim' first, or set ISAACSIM_DIR." >&2
  exit 1
fi

case "$MODE" in
  apply|--revert) ;;
  *) echo "usage: $0 [--revert]" >&2; exit 2 ;;
esac

changed=0
while IFS= read -r -d '' so; do
  base="${so%.cpython-312-*.so}"
  py="$base.py"
  if [[ "$MODE" == "apply" && -f "$py" ]]; then
    mv -f "$py" "$py.disabled"
    echo "disabled: ${py#"$ISAACSIM_DIR"/}"
    changed=$((changed + 1))
  elif [[ "$MODE" == "--revert" && -f "$py.disabled" && ! -e "$py" ]]; then
    mv "$py.disabled" "$py"
    echo "restored: ${py#"$ISAACSIM_DIR"/}"
    changed=$((changed + 1))
  fi
done < <(find "$ISAACSIM_DIR" -type f -name '*.cpython-312-*.so' -print0)

if [[ "$MODE" == "apply" ]]; then
  echo "done: $changed file(s) disabled"
else
  echo "done: $changed file(s) restored"
fi
