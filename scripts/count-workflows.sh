#!/usr/bin/env bash
# Count top-level Claude Code kernel workflows and refresh badges/workflows.json.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BADGE="$ROOT/badges/workflows.json"

count=0
for dir in "$ROOT"/*/; do
  name="$(basename "$dir")"
  case "$name" in
    _*|scripts|experiments|Awesome-LLM-Kernel|output|outputs|kernel_cache|candidates|__pycache__)
      continue
      ;;
  esac
  if compgen -G "${dir}"*.js > /dev/null; then
    count=$((count + 1))
  fi
done

mkdir -p "$(dirname "$BADGE")"
cat > "$BADGE" <<EOF
{
  "schemaVersion": 1,
  "label": "workflows",
  "message": "${count}",
  "color": "7C3AED"
}
EOF

echo "Workflow count: ${count} (wrote ${BADGE})"
