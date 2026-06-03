#!/usr/bin/env bash
# Install Awesome-Kernel-Workflows into Claude Code global workflows (~/.claude/workflows).
#
# Usage:
#   ./scripts/install-global-workflows.sh              # symlinks (default)
#   ./scripts/install-global-workflows.sh --copy     # copy files instead
#   WORKFLOW_DIR=/path/to/Awesome-Kernel-Workflows ./scripts/install-global-workflows.sh
#
# Claude Code discovers ~/.claude/workflows/*.js in every project.
# See: https://code.claude.com/docs/en/workflows

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKFLOW_DIR="${WORKFLOW_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TARGET_DIR="${CLAUDE_WORKFLOWS_DIR:-$HOME/.claude/workflows}"
MODE="symlink"

if [[ "${1:-}" == "--copy" ]]; then
  MODE="copy"
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,12p' "$0"
  exit 0
fi

is_workflow_dir() {
  local d="$1"
  find "$d" -maxdepth 2 \( -name '*-kernel-*.js' -o -name '*-operator-*.js' \) 2>/dev/null | grep -q .
}

if ! is_workflow_dir "$WORKFLOW_DIR"; then
  echo "错误: 未找到 workflow 目录: $WORKFLOW_DIR" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"

mapfile -t JS_FILES < <(find "$WORKFLOW_DIR" -maxdepth 2 -name '*.js' \
  -not -path '*/_templates/*' \
  -not -path '*/_tools/*' \
  -not -path '*/_meta/*' \
  -not -path '*/scripts/*' \
  -not -path '*/experiments/*' \
  -not -path '*/kernel_cache/*' \
  | sort)

if [[ ${#JS_FILES[@]} -eq 0 ]]; then
  echo "错误: $WORKFLOW_DIR 下没有可安装的 workflow .js" >&2
  exit 1
fi

installed=0
for src in "${JS_FILES[@]}"; do
  base="$(basename "$src")"
  dest="$TARGET_DIR/$base"

  if [[ -e "$dest" && ! -L "$dest" && "$MODE" == "symlink" ]]; then
    echo "跳过 (已存在普通文件): $dest" >&2
    continue
  fi

  rm -f "$dest"
  if [[ "$MODE" == "copy" ]]; then
    cp "$src" "$dest"
  else
    ln -sf "$src" "$dest"
  fi
  echo "  $base -> $dest"
  installed=$((installed + 1))
done

echo ""
echo "已安装 $installed 个全局 workflow 到: $TARGET_DIR"
echo "在 Claude Code 中可用 /workflows 浏览，或用 meta.name 调用，例如:"
echo "  Workflow({ name: 'ako4x-kernel-optimizer', args: { ... } })"
echo ""
echo "源目录: $WORKFLOW_DIR"
echo "更新源仓库后，symlink 模式会自动生效；copy 模式请重新运行本脚本。"
