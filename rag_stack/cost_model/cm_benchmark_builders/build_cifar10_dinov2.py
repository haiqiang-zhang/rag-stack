"""Build the cifar10 + dinov2-small ANN dataset for the HNSW cost-model benchmark.

Embeds CIFAR-10 images (uoft-cs/cifar10) with facebook/dinov2-small (ViT-S/14,
CLS token → 384-d), L2-normalized, saved in the ann-benchmarks hdf5 layout
(`train` = base vectors, `test` = query vectors) to mirror glove-100-angular.hdf5.

    conda run -n rag-stack python -m rag_stack.cost_model.cm_benchmark_builders.build_cifar10_dinov2
    → datasets/cm_benchmark/cifar10-dinov2small-384.hdf5  (train 50000×384, test 10000×384)
"""
import argparse
import os
import h5py
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "datasets/cm_benchmark/cifar10-dinov2small-384.hdf5")
MODEL = "facebook/dinov2-small"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def embed(images, proc, model, bs=256):
    """List of PIL images → (N, 384) L2-normalized float32 (dinov2 CLS token)."""
    out = []
    for i in range(0, len(images), bs):
        batch = images[i:i + bs]
        inp = proc(images=batch, return_tensors="pt").to(DEV)
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=(DEV == "cuda")):
            o = model(**inp)
        out.append(o.last_hidden_state[:, 0].float().cpu().numpy())   # [CLS] = whole-image repr
        if (i // bs) % 20 == 0:
            print(f"    {i + len(batch)}/{len(images)}", flush=True)
    x = np.concatenate(out).astype(np.float32)
    x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)            # cosine/angular → unit norm
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    print(f"loading {MODEL} on {DEV} ...", flush=True)
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(DEV).eval()

    print("loading cifar10 ...", flush=True)
    ds = load_dataset("uoft-cs/cifar10")
    img_col = "img" if "img" in ds["train"].column_names else "image"

    print("embedding train (base) ...", flush=True)
    train = embed(ds["train"][img_col], proc, model)
    print("embedding test (query) ...", flush=True)
    test = embed(ds["test"][img_col], proc, model)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)
        f.attrs["embedding_model"] = "dinov2_small"
        f.attrs["dataset_name"] = "cifar10"
        f.attrs["dim"] = train.shape[1]
    print(f"wrote {args.out}: train {train.shape}, test {test.shape}", flush=True)


if __name__ == "__main__":
    main()
