#!/usr/bin/env bash
# Sync WarpSpeed to the GPU node for real acceptance testing.
#   tests/node/sync_to_node.sh [host] [remote_dir]
set -eu
HOST=${1:-H100-lsh}
DEST=${2:-warpspeed-accept}
HERE=$(cd "$(dirname "$0")/../.." && pwd)

ssh "$HOST" "mkdir -p $DEST"
rsync -az --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.ncu-rep' \
  "$HERE/" "$HOST:$DEST/WarpSpeed/"
echo "synced -> $HOST:$DEST/WarpSpeed"
echo "run:    ssh $HOST '$DEST/WarpSpeed/tests/node/run_node_tests.sh'"
