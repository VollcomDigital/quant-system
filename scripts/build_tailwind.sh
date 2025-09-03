#!/usr/bin/env bash
set -euo pipefail

INPUT="src/reporting/tailwind.input.css"
CONFIG="tailwind.config.js"
OUTPUT_DIR="exports/reports/assets"
OUTPUT="$OUTPUT_DIR/tailwind.min.css"

if ! command -v npx >/dev/null 2>&1; then
  echo "npx is required. Please install Node.js (>=18) and try again." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Building Tailwind CSS → $OUTPUT"
npx tailwindcss -c "$CONFIG" -i "$INPUT" -o "$OUTPUT" --minify
echo "Done. Set TAILWIND_CSS_HREF=$OUTPUT to use the local file."
