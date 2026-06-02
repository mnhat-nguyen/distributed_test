"""
ResNet + PyTorch DDP — Ring AllReduce
4 nodes x 1 GPU = world_size 4

Ring AllReduce flow:
  Each GPU holds a shard of the gradient.
  Gradients travel around the ring until every GPU has the full sum.

  GPU0 → GPU1 → GPU2 → GPU3
   ↑________________________↓

Environment variables (set by torchrun):
  MASTER_ADDR   IP of rank-0 node
  MASTER_PORT   free port on rank-0 (rendezvous)
  WORLD_SIZE    total GPUs (4)
  RANK          global rank 0-3
  LOCAL_RANK    local rank within node (always 0, one GPU per node)
"""

import os
import time
import socket
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    """Rank-0 only."""
    if dist.get_rank() == 0:
        print(f"[{now()}][rank 0] {msg}", flush=True)

def log_all(msg):
    """Every rank — used for connection events."""
    print(f"[{now()}][rank {dist.get_rank()} | {socket.gethostname()}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Distributed init
# ─────────────────────────────────────────────────────────────────────────────

def setup():
    """
    Init NCCL process group.

    NCCL automatically selects Ring AllReduce as its collective algorithm —
    it is the default and most efficient strategy for multi-node GPU training.

    Extra NCCL env vars that enforce Ring AllReduce behavior:
      NCCL_ALGO=Ring         - explicitly pick Ring over Tree
      NCCL_PROTO=Simple      - use Simple protocol (best for large gradients)
    These are set in launch_no_k8s.sh and pytorchjob.yaml.
    """
    rank      = int(os.environ.get("RANK",       0))
    world     = int(os.environ.get("WORLD_SIZE", 1))
    master    = os.environ.get("MASTER_ADDR", "?")
    port      = os.environ.get("MASTER_PORT", "?")
    host      = socket.gethostname()

    print(
        f"[{now()}][rank {rank} | {host}] "
        f"Connecting → {master}:{port}  "
        f"(waiting for {world} nodes…)",
        flush=True,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device_id = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device_id)   # NCCL = Ring AllReduce under the hood

    gpu = torch.cuda.get_device_name(local_rank)
    log_all(f"✔ Connected  GPU={gpu}")

    # Barrier: rank-0 prints only after ALL nodes are in
    dist.barrier()
    if dist.get_rank() == 0:
        print(
            f"\n[{now()}] {'='*62}\n"
            f"  ALL {dist.get_world_size()} NODES CONNECTED — Ring AllReduce READY\n"
            f"{'='*62}\n",
            flush=True,
        )

    return local_rank


def cleanup():
    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, arch: str, device, local_rank: int) -> nn.Module:
    model = getattr(models, arch)(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # SyncBatchNorm: BN statistics gathered across all 4 GPUs via AllReduce
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP wraps the model — gradient AllReduce happens automatically
    # after each backward() call, using Ring AllReduce via NCCL
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Data  (CIFAR-10 — swap for your own dataset)
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(args):
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = torchvision.datasets.CIFAR10(args.data_dir, train=True,  download=True, transform=train_tf)
    val_ds   = torchvision.datasets.CIFAR10(args.data_dir, train=False, download=True, transform=val_tf)

    # DistributedSampler shards dataset evenly — each GPU sees 1/4 of the data
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, sampler=val_sampler,
                              num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader, train_sampler


# ─────────────────────────────────────────────────────────────────────────────
# Train / Validate
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, sampler, optimizer, criterion, scaler, device, epoch):
    model.train()
    sampler.set_epoch(epoch)   # different shuffle every epoch

    loss_sum = torch.zeros(1, device=device)
    correct  = torch.zeros(1, device=device)
    total    = torch.zeros(1, device=device)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            out  = model(imgs)
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        # ↑ DDP automatically triggers Ring AllReduce here to sync gradients
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.detach()
        correct  += (out.argmax(1) == labels).sum()
        total    += labels.size(0)

    # Aggregate metrics across all 4 GPUs
    for t in (loss_sum, correct, total):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    return (loss_sum / dist.get_world_size() / len(loader)).item(), \
           (correct / total * 100).item()


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    loss_sum = torch.zeros(1, device=device)
    correct  = torch.zeros(1, device=device)
    total    = torch.zeros(1, device=device)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda"):
            out  = model(imgs)
            loss = criterion(out, labels)

        loss_sum += loss
        correct  += (out.argmax(1) == labels).sum()
        total    += labels.size(0)

    for t in (loss_sum, correct, total):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    return (loss_sum / dist.get_world_size() / len(loader)).item(), \
           (correct / total * 100).item()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_ckpt(state, path):
    if dist.get_rank() == 0:
        torch.save(state, path)
        log(f"Checkpoint saved → {path}")


def load_ckpt(path, model, optimizer, scheduler):
    if not os.path.isfile(path):
        return 0
    ckpt = torch.load(path, map_location="cpu")
    sd   = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.module.load_state_dict(sd)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    log(f"Resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",         default="resnet50")
    parser.add_argument("--num_classes",  type=int,   default=10)
    parser.add_argument("--epochs",       type=int,   default=90)
    parser.add_argument("--batch_size",   type=int,   default=64,  help="per-GPU batch size")
    parser.add_argument("--lr",           type=float, default=0.1, help="base LR (scaled by world_size)")
    parser.add_argument("--momentum",     type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--data_dir",     default="./data")
    parser.add_argument("--output_dir",   default="./checkpoints")
    parser.add_argument("--resume",       default="")
    parser.add_argument("--save_every",   type=int,   default=10)
    args = parser.parse_args()

    # ── Init ──────────────────────────────────────────────────────────────────
    local_rank = setup()
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device(f"cuda:{local_rank}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.num_classes, args.arch, device, local_rank)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Linear scaling rule: larger global batch → larger LR
    scaled_lr = args.lr * world_size   # 0.1 * 4 = 0.4
    optimizer = optim.SGD(model.parameters(), lr=scaled_lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    warmup   = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 5)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[5])

    criterion = nn.CrossEntropyLoss().to(device)
    scaler    = torch.amp.GradScaler('cuda')

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, train_sampler = build_loaders(args)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = load_ckpt(args.resume, model, optimizer, scheduler) if args.resume else 0

    # ── Training loop ─────────────────────────────────────────────────────────
    log(
        f"arch={args.arch}  epochs={args.epochs}  "
        f"per-GPU batch={args.batch_size}  global batch={args.batch_size * world_size}  "
        f"lr={scaled_lr:.4f}  AllReduce=Ring(NCCL)"
    )

    t_total    = time.time()
    epoch_times = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, train_sampler,
            optimizer, criterion, scaler, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        avg_t    = sum(epoch_times) / len(epoch_times)
        eta      = str(datetime.timedelta(seconds=int(avg_t * (args.epochs - epoch - 1))))
        pct      = (epoch + 1) / args.epochs
        bar      = "█" * int(20 * pct) + "░" * (20 - int(20 * pct))

        log(
            f"[{bar}] {epoch+1}/{args.epochs} ({pct*100:.0f}%)  "
            f"train {train_loss:.4f}/{train_acc:.1f}%  "
            f"val {val_loss:.4f}/{val_acc:.1f}%  "
            f"lr={scheduler.get_last_lr()[0]:.5f}  "
            f"epoch={elapsed:.0f}s  ETA={eta}"
        )

        if (epoch + 1) % args.save_every == 0:
            save_ckpt(
                {"epoch": epoch + 1, "arch": args.arch,
                 "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict()},
                os.path.join(args.output_dir, f"ckpt_epoch{epoch+1}.pt"),
            )

    # ── Done ──────────────────────────────────────────────────────────────────
    total_time = str(datetime.timedelta(seconds=int(time.time() - t_total)))
    save_ckpt({"epoch": args.epochs, "arch": args.arch, "model": model.state_dict()},
              os.path.join(args.output_dir, "ckpt_final.pt"))

    if dist.get_rank() == 0:
        with open(os.path.join(args.output_dir, "TRAINING_DONE.txt"), "w") as f:
            f.write(f"finished : {now()}\ntotal_time : {total_time}\n")

    log(f"\n{'='*62}\n  TRAINING DONE  total={total_time}\n{'='*62}")
    cleanup()


if __name__ == "__main__":
    main()
