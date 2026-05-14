#!/usr/bin/env bash
# Pull a Modal sweep result from the `pr-runs` volume, one subdir at a time.
#
# Why: `modal volume get pr-runs /<dir> ./runs/<dir> --force` raises
# "[Errno 21] Is a directory" whenever the local target already exists
# AND the source is a directory — a long-standing CLI quirk. The
# work-around is to fetch each leaf subdir individually so each `get`
# call sees a non-existing local target.
#
# Usage:
#   scripts/pull_sweep.sh                 # defaults: VOL=pr-runs SWEEP=cf_sweep
#   SWEEP=cf_sweep_v2 scripts/pull_sweep.sh
#   VOL=pr-runs SWEEP=cf_sweep scripts/pull_sweep.sh
#
# Then aggregate locally:
#   python train/compare_sweep.py runs/cf_sweep -o reports/cf_sweep.md
set -euo pipefail

VOL="${VOL:-pr-runs}"
SWEEP="${SWEEP:-cf_sweep}"
LOCAL_DIR="${LOCAL_DIR:-runs/${SWEEP}}"

CFS=("1.00" "1.25" "1.50" "2.00")

# Ensure local dir exists but is empty enough that per-subdir `get` works.
mkdir -p "${LOCAL_DIR}"

# Discover what's actually on the volume (so the script tolerates a partial sweep).
echo "[pull_sweep] listing /${SWEEP} on volume ${VOL}…"
present=$(modal volume ls "${VOL}" "/${SWEEP}" 2>/dev/null \
          | awk 'NR>1 {print $2}' \
          | sed 's:/$::' \
          || true)

if [[ -z "${present}" ]]; then
  echo "[pull_sweep] no entries under /${SWEEP} — is the path right?" >&2
  exit 1
fi

# Pull every entry. Files come back as files; directories trigger the
# "Is a directory" bug if the local target exists, so we always remove
# the local copy first.
while IFS= read -r entry; do
  [[ -z "${entry}" ]] && continue
  remote="/${SWEEP}/${entry}"
  local="${LOCAL_DIR}/${entry}"
  echo "[pull_sweep] ${remote}  →  ${local}"
  rm -rf "${local}"
  modal volume get "${VOL}" "${remote}" "${local}" --force
done <<< "${present}"

echo "[pull_sweep] done. Aggregate with:"
echo "  python train/compare_sweep.py ${LOCAL_DIR} -o reports/${SWEEP}.md"
