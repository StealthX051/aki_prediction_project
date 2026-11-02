#!/usr/bin/env bash
# Replay captured Linux binaries into a new Conda prefix, then verify offline.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
conda_exe="${AKI_CONDA_EXE:-${CONDA_EXE:-conda}}"
: "${AKI_REBUILD_PREFIX:?Set AKI_REBUILD_PREFIX to a new environment path}"
: "${AKI_WHEELHOUSE:?Set AKI_WHEELHOUSE to the captured Linux CPython 3.12 wheels}"
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  echo 'These binary locks require Linux x86_64.' >&2
  exit 1
fi
destination=$(realpath -m -- "$AKI_REBUILD_PREFIX")
if [[ -e "$AKI_REBUILD_PREFIX" || -L "$AKI_REBUILD_PREFIX" || -e "$destination" || -L "$destination" ]]; then
  echo "Refusing existing rebuild path: $AKI_REBUILD_PREFIX" >&2
  exit 1
fi
if [[ ! -d "$AKI_WHEELHOUSE" ]]; then
  echo "Wheelhouse not found: $AKI_WHEELHOUSE" >&2
  exit 1
fi
wheelhouse=$(cd -- "$AKI_WHEELHOUSE" && pwd)
evidence=$(mktemp -d /tmp/aki-cpu-rebuild.XXXXXXXX)
mkdir -- "$evidence/tmp"
export TMPDIR="$evidence/tmp"
echo "Rebuild evidence: $evidence"
export PYTHONNOUSERSITE=1
"$conda_exe" create --yes --prefix "$destination" \
  --file "$repo/locks/conda-linux-64.explicit.txt" 2>&1 | tee "$evidence/conda-create.log"
"$destination/bin/python" -m pip --isolated install --no-index --only-binary=:all: \
  --find-links "$wheelhouse" --require-hashes \
  -r "$repo/locks/requirements-linux-64.wheels.txt" 2>&1 | tee "$evidence/pip-install.log"
# Record and compare exact Conda URLs/builds/checksums, excluding export comments.
"$conda_exe" list --prefix "$destination" --explicit --sha256 > "$evidence/conda-explicit.txt"
diff -u <(sed '/^#/d; /^$/d' "$repo/locks/conda-linux-64.explicit.txt" | sort) \
  <(sed '/^#/d; /^$/d' "$evidence/conda-explicit.txt" | sort)
# Activate instead of relying on a particular Miniforge installation path.
eval "$("$conda_exe" shell.bash hook)"
conda activate "$destination"
AKI_VERIFY_ROOT="$evidence/checks" bash "$repo/scripts/cpu/verify-offline.sh"
echo "Fresh lock rebuild passed: $destination"
echo "Evidence: $evidence"
