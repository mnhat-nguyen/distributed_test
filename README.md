# ResNet DDP — 4 Nodes × 1 GPU

Distributed ResNet training with **PyTorch DDP** across **4 nodes, each with 1 GPU**
(world size = 4). Supports two deployment modes:

| Mode | File(s) |
|------|---------|
| **No Kubernetes** (bare metal / VMs) | `launch_no_k8s.sh` |
| **Kubernetes** (Kubeflow Training Operator) | `k8s/pytorchjob.yaml`, `k8s/pvc.yaml` |

---

## Repository layout

```
resnet-ddp/
├── train.py               # DDP training script (shared by both modes)
├── launch_no_k8s.sh       # Launcher for bare-metal / VM clusters
├── Dockerfile             # Container image for K8s deployment
├── k8s/
│   ├── pytorchjob.yaml    # Kubeflow PyTorchJob (4 nodes × 1 GPU)
│   └── pvc.yaml           # PersistentVolumeClaims for data + checkpoints
└── README.md
```

---

## Option A — No Kubernetes (bare metal / VM)

### Requirements
- Python ≥ 3.9, PyTorch ≥ 2.0, torchvision
- All 4 nodes can reach each other over TCP (port 29500 open)
- Shared filesystem **or** dataset pre-copied to each node under `./data`

### Steps

1. **Copy files to all 4 nodes** (same path on each):
   ```bash
   scp train.py launch_no_k8s.sh user@node{1,2,3}:~/resnet-ddp/
   ```

2. **Run on every node** — each gets a unique `NODE_RANK`:
   ```bash
   # Node 0 (master — must start first or simultaneously)
   MASTER_ADDR=10.0.0.1 NODE_RANK=0 bash launch_no_k8s.sh

   # Node 1
   MASTER_ADDR=10.0.0.1 NODE_RANK=1 bash launch_no_k8s.sh

   # Node 2
   MASTER_ADDR=10.0.0.1 NODE_RANK=2 bash launch_no_k8s.sh

   # Node 3
   MASTER_ADDR=10.0.0.1 NODE_RANK=3 bash launch_no_k8s.sh
   ```
   Training begins once **all 4 nodes** complete the rendezvous.

3. **Resume from a checkpoint**:
   ```bash
   NODE_RANK=0 bash launch_no_k8s.sh --resume ./checkpoints/checkpoint_epoch50.pt
   ```

### Tip — automate with pdsh / parallel-ssh
```bash
pdsh -w node[0-3] "cd ~/resnet-ddp && MASTER_ADDR=10.0.0.1 NODE_RANK=%h bash launch_no_k8s.sh"
```

---

## Option B — Kubernetes (Kubeflow Training Operator)

### Requirements
- Kubernetes cluster with GPU nodes (1 GPU per node × 4 nodes)
- [Kubeflow Training Operator](https://github.com/kubeflow/training-operator) installed
- A shared storage class that supports `ReadWriteMany` (NFS, GCS Fuse, EFS, Azure Files)

### Steps

1. **Install the Training Operator** (if not already):
   ```bash
   kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone"
   ```

2. **Build and push the Docker image**:
   ```bash
   docker build -t my-registry/resnet-ddp:latest .
   docker push my-registry/resnet-ddp:latest
   ```
   Update the `image:` field in `k8s/pytorchjob.yaml`.

3. **Create PVCs** (shared data + checkpoint storage):
   ```bash
   kubectl apply -f k8s/pvc.yaml
   ```

4. **Submit the training job**:
   ```bash
   kubectl apply -f k8s/pytorchjob.yaml
   ```

5. **Monitor**:
   ```bash
   # Job status
   kubectl get pytorchjob resnet-ddp-job -n kubeflow

   # Logs from master (rank 0)
   kubectl logs -n kubeflow -l job-name=resnet-ddp-job -c pytorch -f

   # All pods
   kubectl get pods -n kubeflow -l app=resnet-ddp
   ```

6. **Delete**:
   ```bash
   kubectl delete -f k8s/pytorchjob.yaml
   ```

---

## Key design decisions

| Feature | Detail |
|---------|--------|
| **Backend** | NCCL (optimal GPU↔GPU comms, supports InfiniBand & RoCE) |
| **Mixed precision** | `torch.autocast` + `GradScaler` (AMP) for faster training |
| **SyncBatchNorm** | BatchNorm statistics synced across all 4 GPUs |
| **LR scaling** | Linear scaling rule: `lr × world_size` (0.1 × 4 = 0.4) |
| **LR schedule** | 5-epoch linear warm-up → cosine annealing |
| **Checkpointing** | Only rank-0 writes to avoid filesystem races |
| **Shuffling** | `sampler.set_epoch(epoch)` ensures different shuffle every epoch |

---

## Hyperparameter reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--arch` | `resnet50` | Any torchvision ResNet (`resnet18`, `resnet101`, etc.) |
| `--epochs` | `90` | Total training epochs |
| `--batch_size` | `64` | **Per-GPU** batch (global = 64 × 4 = 256) |
| `--lr` | `0.1` | Base LR (auto-scaled to `0.1 × 4 = 0.4`) |
| `--num_classes` | `10` | Output classes (10 for CIFAR-10) |
| `--data_dir` | `./data` | Dataset root |
| `--output_dir` | `./checkpoints` | Where checkpoints are saved |
| `--resume` | `` | Path to checkpoint to resume from |
| `--save_every` | `10` | Save checkpoint every N epochs |

---

## Swap out CIFAR-10 for your own dataset

Replace the `build_dataloaders()` function in `train.py`.
The only requirement is that the dataset returns `(image_tensor, label)` pairs
and can be wrapped with `DistributedSampler`.

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, root, split, transform=None): ...
    def __len__(self): ...
    def __getitem__(self, idx): ...  # return (image_tensor, label_int)
```
"# distributed_test" 
