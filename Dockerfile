# =============================================================================
# Dockerfile
#
# Build:   docker build -t my-registry/resnet-ddp:latest .
# Push:    docker push my-registry/resnet-ddp:latest
#
# Then update the `image:` field in k8s/pytorchjob.yaml.
# =============================================================================

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Install torchvision + misc utilities
RUN pip install --no-cache-dir \
        torchvision==0.18.0 \
        tqdm \
    && rm -rf /root/.cache/pip

WORKDIR /workspace

# Copy training script into the image
COPY train.py /workspace/train.py

# Create directories that will be overridden by PVC mounts in K8s
RUN mkdir -p /data /checkpoints

# Default: show help (actual command is supplied by the K8s job spec / launch script)
ENTRYPOINT ["python", "train.py"]
CMD ["--help"]
