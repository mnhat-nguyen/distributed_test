#!/usr/bin/env bash
# =============================================================================
# launch_no_k8s.sh  –  Run 4-node x 1-GPU DDP without Kubernetes
#
# USAGE (run this script on EVERY node):
#   export MASTER_ADDR=<node-0-ip>
#   export MASTER_PORT=29500
#   export NODE_RANK=<0|1|2|3>          # unique per node
#   bash launch_no_k8s.sh [extra train.py args]
#
# Or pass everything inline:
#   MASTER_ADDR=10.0.0.1 NODE_RANK=0 bash launch_no_k8s.sh
#   MASTER_ADDR=10.0.0.1 NODE_RANK=1 bash launch_no_k8s.sh   ← on node 1
#   ...
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
MASTER_ADDR="${MASTER_ADDR:-192.168.0.11}"   # IP of node 0 (all nodes must reach it)
MASTER_PORT="${MASTER_PORT:-20011}"
NNODES=4                                  # total number of nodes
NPROC_PER_NODE=1                          # GPUs per node
NODE_RANK="${NODE_RANK:-0}"               # 0-3, must be unique per node

# ── Optional: virtualenv / conda activation ───────────────────────────────────
# source /opt/venv/bin/activate
# conda activate myenv

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! command -v torchrun &>/dev/null; then
    echo "[ERROR] torchrun not found. Install PyTorch >= 1.10."
    exit 1
fi

if [[ "${NODE_RANK}" == "0" ]]; then
    echo "======================================================"
    echo "  ResNet DDP — 4 nodes x 1 GPU (no Kubernetes)"
    echo "  MASTER_ADDR : ${MASTER_ADDR}:${MASTER_PORT}"
    echo "  This node   : rank ${NODE_RANK}"
    echo "======================================================"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
# torchrun handles:
#   - spawning 1 process per GPU
#   - setting LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
#   - rendezvous via c10d (TCP)

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    train.py \
        --arch        resnet50 \
        --epochs      90 \
        --batch_size  64 \
        --lr          0.1 \
        --data_dir    ./data \
        --output_dir  ./checkpoints \
        "$@"          # pass any extra args from CLI

# =============================================================================
# QUICK-START CHEAT SHEET
#
# 1. Copy train.py + this script to ALL 4 nodes (same path).
#
# 2. Ensure nodes can reach each other:
#      ping <MASTER_ADDR>
#      nc -zv <MASTER_ADDR> 20011    # port must be open
#
# 3. On each node, set NODE_RANK and run:
#      Node 0:  MASTER_ADDR=192.168.0.11 NODE_RANK=0 bash launch_no_k8s.sh
#      Node 1:  MASTER_ADDR=192.168.0.11 NODE_RANK=1 bash launch_no_k8s.sh
#      Node 2:  MASTER_ADDR=192.168.0.11 NODE_RANK=2 bash launch_no_k8s.sh
#      Node 3:  MASTER_ADDR=192.168.0.11 NODE_RANK=3 bash launch_no_k8s.sh
#
# 4. Training starts once ALL nodes have connected to the rendezvous.
#
# RESUME from checkpoint:
#      NODE_RANK=0 bash launch_no_k8s.sh --resume ./checkpoints/checkpoint_epoch50.pt
# =============================================================================
