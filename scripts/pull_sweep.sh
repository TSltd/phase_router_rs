#!/usr/bin/env bash
# Pull a Modal sweep result from the `pr-runs` volume — file by file.
#
# Why file-by-file: `modal volume get` on a *directory* is unreliable
# across CLI versions. Two pathologies we hit in practice:
#   1. "[Errno 21] Is a directory" when the local target already exists
#      as a directory and you re-fetch — the CLI tries to overwrite a
#      dir with a single file handle.
#   2. With newer CLIs the request "succeeds" but the directory is
#      collapsed into a single binary blob (turned out to be the
#      remote `model.pt` renamed to the parent directory's name),
#      and `metrics.jsonl` is silently dropped. `compare_sweep.py`
#      then can't find any data.
#
# Fetching individual *files* always works. So this script:
#   • lists the contents of /<sweep>/ on the volume,
#   • for each subdir, lists its files,
#   • pulls each file to a path that mirrors the volume layout,
#     creating local dirs on demand.
#
# Usage:
#   scripts/pull_sweep.sh                         # defaults: VOL=pr-runs SWEEP=cf_sweep
#   SWEEP=cf_sweep_v2 scripts/pull_sweep.sh
#   VOL=pr-runs SWEEP=cf_sweep_v2 LOCAL_DIR=runs/cf_sweep_v2 scripts/pull_sweep.sh
#   SKIP_MODEL=1 SWEEP=cf_sweep_v2 scripts/pull_sweep.sh   # skip model.pt (much faster)
#
# After it finishes, aggregate locally:
#   python train/compare_sweep.py runs/<SWEEP> -o reports/<SWEEP>.md

set -euo pipefail

VOL="${VOL:-pr-runs}"
SWEEP="${SWEEP:-cf_sweep}"
LOCAL_DIR="${LOCAL_DIR:-runs/${SWEEP}}"
# When SKIP_MODEL=1, don't pull `model.pt` files. compare_sweep.py only
# needs `metrics.jsonl`, and the checkpoints are ~123 MB each — pulling
# them all turns a 10-second job into a 5-minute one.
SKIP_MODEL="${SKIP_MODEL:-0}"


echo "[pull_sweep] vol=${VOL} sweep=/${SWEEP} local=${LOCAL_DIR}"

# Wipe any stale local state for this sweep so we don't end up with
# half-pulled / collapsed-blob mixed contents (e.g. from an earlier
# `modal volume get` of a directory). Comment this out if you want
# resume-safe behaviour, but for paper-grade reproducibility a clean
# pull is safer.
if [[ -d "${LOCAL_DIR}" || -e "${LOCAL_DIR}" ]]; then
  echo "[pull_sweep] clearing stale local copy at ${LOCAL_DIR}"
  rm -rf -- "${LOCAL_DIR}"
fi
mkdir -p "${LOCAL_DIR}"

# `modal volume ls` prints one path per line, e.g. `cf_sweep_v2/phase_cf1.00`.
# Some versions add a header — strip anything that doesn't look like our
# sweep prefix.
list_remote() {
  local path="$1"
  modal volume ls "${VOL}" "${path}" 2>/dev/null \
    | awk -v p="${SWEEP}" '$0 ~ ("^" p "/") {print $0}'
}

# Top-level entries inside /<SWEEP>/.
mapfile -t TOP_ENTRIES < <(list_remote "/${SWEEP}")
if (( ${#TOP_ENTRIES[@]} == 0 )); then
  echo "[pull_sweep] nothing under /${SWEEP} on ${VOL} — wrong name?" >&2
  exit 1
fi

# For each entry: if it's a file (e.g. sweep.md), pull directly.
# If it's a directory (e.g. phase_cf1.00), recurse one level and pull each file.
# We can tell them apart by trying to `ls` the entry — directories yield
# children, files yield empty.
for entry in "${TOP_ENTRIES[@]}"; do
  # entry is `<SWEEP>/<name>`; strip the prefix for clarity.
  name="${entry#${SWEEP}/}"
  remote="/${entry}"

  # Probe: is this a directory?
  mapfile -t children < <(list_remote "/${entry}")
  if (( ${#children[@]} == 0 )); then
    # Treat as a file. Single-file `modal volume get` is rock-solid.
    local_path="${LOCAL_DIR}/${name}"
    echo "[pull_sweep] file  ${remote}  →  ${local_path}"
    mkdir -p "$(dirname -- "${local_path}")"
    modal volume get "${VOL}" "${remote}" "${local_path}" --force
  else
    # Directory: pull each child individually.
    mkdir -p "${LOCAL_DIR}/${name}"
    for child in "${children[@]}"; do
      # `child` is `<SWEEP>/<name>/<file>` (or possibly deeper, but our
      # sweep layout is flat — one level of files under each subdir).
      child_name="${child#${SWEEP}/${name}/}"
      if [[ "${SKIP_MODEL}" == "1" && "${child_name}" == "model.pt" ]]; then
        echo "[pull_sweep] skip  /${child}  (SKIP_MODEL=1)"
        continue
      fi
      remote_file="/${child}"
      local_file="${LOCAL_DIR}/${name}/${child_name}"
      echo "[pull_sweep] file  ${remote_file}  →  ${local_file}"
      mkdir -p "$(dirname -- "${local_file}")"
      modal volume get "${VOL}" "${remote_file}" "${local_file}" --force
    done

  fi
done

echo "[pull_sweep] done."
echo "[pull_sweep] aggregate with:"
echo "  python train/compare_sweep.py ${LOCAL_DIR} -o reports/${SWEEP}.md"
