#!/usr/bin/env bash

set -euo pipefail

# Allow optional CLI args to limit which configs run; default to every *_baseline yml/yaml.
declare -a configs=("$@")
if [ "${#configs[@]}" -eq 0 ]; then
  shopt -s nullglob
  configs=(config/baseline/*_baseline.yml config/baseline/*_baseline.yaml)
  shopt -u nullglob
fi

if [ "${#configs[@]}" -eq 0 ]; then
  echo "No *_baseline.{yml,yaml} configs found under config/baseline" >&2
  exit 1
fi

counter=1
for cfg in "${configs[@]}"; do
  if [ ! -f "$cfg" ]; then
    echo "Skipping missing config $cfg" >&2
    ((counter++))
    continue
  fi
  base_name="$(basename "$cfg")"
  run_id="${base_name%.*}-$(date +%Y%m%d%H%M%S)-${counter}"
  echo
  echo ">>> Running baseline config: $cfg"
  echo ">>> RUN_ID=$run_id"
  if ! RUN_ID="$run_id" poetry run python -m src.main run --config "$cfg"; then
    echo "!!! Failed: $cfg (see logs above). Continuing with next baseline." >&2
  fi
  ((counter++))
done

echo
echo "Completed ${#configs[@]} baseline run(s)."
