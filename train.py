"""
ResNet Distributed Training with PyTorch DDP
4 nodes x 1 GPU each = world_size 4

Env vars consumed (set by torchrun or K8s):
  MASTER_ADDR   - hostname/IP of rank-0 node
  MASTER_PORT   - free port on rank-0 node (default 29500)
  WORLD_SIZE    - total number of processes (4)
  RANK          - global rank of THIS process (0-3)
  LOCAL_RANK    - local rank within the node (always 0 here, 1 GPU/node)
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize the process group using env-var backend (set by torchrun/K8s)."""
    dist.init_process_group(backend="nccl")  # NCCL is optimal for GPU-to-GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def log(msg):
    """Only rank-0 prints to avoid log spam."""
    if is_main_process():
        print(f"[rank 0] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 10, arch: str = "resnet50") -> nn.Module:
    """Return a ResNet model with a custom head for num_classes."""
    constructor = getattr(models, arch)
    model = constructor(weights=None)                  # train from scratch
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Data  (CIFAR-10 as a stand-in – swap for your real dataset)
# ---------------------------------------------------------------------------

def build_dataloaders(args, rank: int, world_size: int):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),              # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform_train
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform_val
    )

    # DistributedSampler shards data across all ranks deterministically
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Training / Validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, sampler, optimizer, criterion, scaler, device, epoch):
    model.train()
    sampler.set_epoch(epoch)   # ensures different shuffling every epoch

    total_loss = torch.tensor(0.0, device=device)
    correct    = torch.tensor(0,   device=device)
    total      = torch.tensor(0,   device=device)

    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):          # mixed precision
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach()
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum()
        total      += labels.size(0)

    # Aggregate metrics across all ranks
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct,    op=dist.ReduceOp.SUM)
    dist.all_reduce(total,      op=dist.ReduceOp.SUM)

    world_size = dist.get_world_size()
    avg_loss   = (total_loss / world_size / len(loader)).item()
    accuracy   = (correct / total * 100).item()
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    correct    = torch.tensor(0,   device=device)
    total      = torch.tensor(0,   device=device)

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        total_loss += loss
        correct    += (outputs.argmax(1) == labels).sum()
        total      += labels.size(0)

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct,    op=dist.ReduceOp.SUM)
    dist.all_reduce(total,      op=dist.ReduceOp.SUM)

    world_size = dist.get_world_size()
    avg_loss   = (total_loss / world_size / len(loader)).item()
    accuracy   = (correct / total * 100).item()
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state, path):
    """Only rank-0 writes checkpoints to avoid file-system races."""
    if is_main_process():
        torch.save(state, path)
        log(f"Checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer, scheduler):
    if not os.path.isfile(path):
        return 0
    checkpoint = torch.load(path, map_location="cpu")
    # Strip the DDP wrapper prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
    model.module.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    log(f"Resumed from epoch {checkpoint['epoch']} (path: {path})")
    return checkpoint["epoch"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ResNet DDP training – 4 nodes x 1 GPU")
    parser.add_argument("--arch",         default="resnet50",  help="torchvision model name")
    parser.add_argument("--num_classes",  type=int, default=10)
    parser.add_argument("--epochs",       type=int, default=90)
    parser.add_argument("--batch_size",   type=int, default=64,
                        help="Per-GPU batch size (global = batch_size * world_size)")
    parser.add_argument("--lr",           type=float, default=0.1,
                        help="Base LR; will be scaled by world_size")
    parser.add_argument("--momentum",     type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--data_dir",     default="./data")
    parser.add_argument("--output_dir",   default="./checkpoints")
    parser.add_argument("--resume",       default="",  help="path to checkpoint to resume from")
    parser.add_argument("--save_every",   type=int, default=10)
    args = parser.parse_args()

    # ── DDP init ──────────────────────────────────────────────────────────
    local_rank  = setup_distributed()
    rank        = dist.get_rank()
    world_size  = dist.get_world_size()
    device      = torch.device(f"cuda:{local_rank}")

    log(f"Distributed init done – world_size={world_size}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(args.num_classes, args.arch).to(device)
    # SyncBatchNorm converts BN layers so statistics are synced across GPUs
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── Optimiser / Scheduler ─────────────────────────────────────────────
    # Linear LR scaling rule: scale lr proportionally to world_size
    scaled_lr = args.lr * world_size
    optimizer = optim.SGD(
        model.parameters(),
        lr=scaled_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    # Cosine annealing with linear warm-up (first 5 epochs)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=5
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - 5
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[5],
    )

    criterion = nn.CrossEntropyLoss().to(device)
    scaler    = torch.cuda.amp.GradScaler()          # AMP loss scaler

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, train_sampler = build_dataloaders(args, rank, world_size)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)

    # ── Training loop ─────────────────────────────────────────────────────
    log(f"Starting training: arch={args.arch}, epochs={args.epochs}, "
        f"per-GPU batch={args.batch_size}, global batch={args.batch_size * world_size}, "
        f"scaled_lr={scaled_lr:.4f}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, train_sampler, optimizer, criterion, scaler, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        log(
            f"Epoch [{epoch+1:3d}/{args.epochs}]  "
            f"train loss={train_loss:.4f}  train acc={train_acc:.2f}%  "
            f"val loss={val_loss:.4f}  val acc={val_acc:.2f}%  "
            f"lr={scheduler.get_last_lr()[0]:.6f}  time={elapsed:.1f}s"
        )

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                {
                    "epoch":     epoch + 1,
                    "arch":      args.arch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt"),
            )

    # Final checkpoint
    save_checkpoint(
        {"epoch": args.epochs, "arch": args.arch, "model": model.state_dict()},
        os.path.join(args.output_dir, "checkpoint_final.pt"),
    )
    log("Training complete.")
    cleanup()


if __name__ == "__main__":
    main()
