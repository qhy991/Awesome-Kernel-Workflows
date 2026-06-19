#!/usr/bin/env bash
# Count routable workflow .js entrypoints (matches KerSor generate-catalog.sh scan)
# and refresh badges/workflows.json.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BADGE="$ROOT/badges/workflows.json"

count=0
while IFS= read -r _; do
  count=$((count + 1))
done < <(
  find "$ROOT" -maxdepth 2 -name '*.js' \
    -not -path '*/_templates/*' \
    -not -path '*/_tools/*' \
    -not -path '*/_substrate/*' \
    -not -path '*/_manifests/*' \
    -not -path '*/_meta/*' \
    -not -path '*/experiments/*' \
    -not -path '*/kernel_cache/*' \
    -not -path '*/candidates/*' \
    -not -path '*/__pycache__/*' \
    -not -path '*/scripts/*' \
    -not -path '*/node_modules/*' \
    | sort
)

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
