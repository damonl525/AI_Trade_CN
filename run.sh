#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# AI_Trade_CN 统一入口
# 自动清除 PYTHONPATH（防止 Hermes 环境污染导致 numpy 冲突）
# 用法: bash run.sh <main.py 的参数>
# 例:   bash run.sh pos
#       bash run.sh signal --tk 2 --lb 20
#       bash run.sh dynbt --all
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHONPATH= uv run python main.py "$@"
