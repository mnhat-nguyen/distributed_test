#!/usr/bin/env bash
# =============================================================================
# launch_no_k8s.sh  —  4 nodes x 1 GPU, Ring AllReduce, no Kubernetes
#
# Run on EVERY node (all 4 at the same time):
#
#   Node 0:  MASTER_ADDR=192.168.0.11 MASTER_PORT=20011 NODE_RANK=0 bash launch_no_k8s.sh
#   Node 1:  MASTER_ADDR=192.168.0.11 MASTER_PORT=20011 NODE_RANK=1 bash launch_no_k8s.sh
#   Node 2:  MASTER_ADDR=192.168.0.11 MASTER_PORT=20011 NODE_RANK=2 bash launch_no_k8s.sh
#   Node 3:  MASTER_ADDR=192.168.0.11 MASTER_PORT=20011 NODE_RANK=3 bash launch_no_k8s.sh
#
# Before running — open ports on ALL nodes:
#   sudo ufw allow 20011/tcp
#   sudo ufw allow 20000:21000/tcp
#   sudo ufw allow 20000:21000/udp
#   sudo ufw reload
# =============================================================================

set -euo pipefail

# ── Settings (override via env vars) ─────────────────────────────────────────
MASTER_ADDR="${MASTER_ADDR:-192.168.0.11}"
MASTER_PORT="${MASTER_PORT:-20011}"
NODE_RANK="${NODE_RANK:-0}"
NNODES=4
NPROC_PER_NODE=1   # 1 GPU per node

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$WORK_DIR/py-ddp"

# ── Ring AllReduce — force NCCL to use Ring algorithm ────────────────────────
export NCCL_ALGO=Ring          # explicitly use Ring AllReduce
export NCCL_PROTO=Simple       # Simple protocol — best for large gradient tensors
export NCCL_DEBUG=WARN         # set to INFO for verbose NCCL connection logs
export NCCL_IB_DISABLE=0       # set to 1 if no InfiniBand
export OMP_NUM_THREADS=4

# ── Activate virtualenv ───────────────────────────────────────────────────────
if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
fi

if [[ "$NODE_RANK" == "0" ]]; then
    echo "============================================================"
    echo "  ResNet DDP — Ring AllReduce (no Kubernetes)"
    echo "  Master   : $MASTER_ADDR:$MASTER_PORT"
    echo "  Rank     : $NODE_RANK / $((NNODES-1))"
    echo "  NCCL_ALGO: $NCCL_ALGO"
    echo "============================================================"
fi

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    "$WORK_DIR/train.py" \
        --arch        resnet50 \
        --epochs      90 \
        --batch_size  64 \
        --lr          0.1 \
        --data_dir    "$WORK_DIR/data" \
        --output_dir  "$WORK_DIR/checkpoints" \
        "$@"
