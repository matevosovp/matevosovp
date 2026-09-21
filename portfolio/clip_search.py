"""Cosine top-k image retrieval and optional OpenCLIP text encoding."""
import argparse
import json
from pathlib import Path
import numpy as np


def nearest_neighbors(vectors, query_index, k=10):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Expected a finite 2D embedding matrix")
    if not 0 <= query_index < len(vectors) or not 1 <= k < len(vectors):
        raise ValueError("Invalid query index or k; require 1 <= k < item count")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero embeddings have no cosine direction")
    normalized = vectors / norms
    scores = normalized @ normalized[query_index]
    scores[query_index] = -np.inf
    candidates = np.argpartition(-scores, k - 1)[:k]
    ordered = candidates[np.lexsort((candidates, -scores[candidates]))]
    return [{"row": int(i), "cosine_similarity": float(scores[i])} for i in ordered]


def encode_brands(brands, checkpoint, batch_size=64):
    import torch
    import open_clip
    if not Path(checkpoint).is_file():
        raise FileNotFoundError("Provide local compatible OpenCLIP weights")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    architecture = "ViT-B-32-quickgelu"
    model, _, _ = open_clip.create_model_and_transforms(architecture, pretrained=str(checkpoint))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(architecture)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(brands), batch_size):
            prompts = [f"a product photo of {b}" for b in brands[start:start + batch_size]]
            output = model.encode_text(tokenizer(prompts).to(device))
            output = torch.nn.functional.normalize(output, dim=-1)
            vectors.append(output.cpu().numpy())
    if not vectors:
        raise ValueError("Provide at least one brand")
    return np.concatenate(vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search")
    search.add_argument("--embeddings", type=Path, required=True)
    search.add_argument("--query-index", type=int, required=True)
    search.add_argument("--top-k", type=int, default=10)
    encode = commands.add_parser("encode-brands")
    encode.add_argument("--brands", type=Path, required=True, help="UTF-8 file, one brand per line")
    encode.add_argument("--checkpoint", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "search":
        print(json.dumps(nearest_neighbors(np.load(args.embeddings, allow_pickle=False), args.query_index, args.top_k), indent=2))
    else:
        brands = [b.strip() for b in args.brands.read_text(encoding="utf-8").splitlines() if b.strip()]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, encode_brands(brands, args.checkpoint))
