FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN pip install --no-cache-dir torchvision==0.18.0 tqdm

WORKDIR /workspace
COPY train.py .
RUN mkdir -p /data /checkpoints

ENTRYPOINT ["python", "train.py"]
CMD ["--help"]
