"""Quick debug for retrieval scores."""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.retrieval import _load_index, normalize_cond
from utils.text_encoder import TextEncoder

text = "A small yellow bird with black wings"
enc = TextEncoder()
q = normalize_cond(enc.encode([text]))
mat, meta = _load_index()
mat = normalize_cond(mat)
sims = torch.matmul(mat, q.squeeze(0))

v, i = torch.topk(sims, 15)
print("Top 15 by cosine:")
for val, idx in zip(v.tolist(), i.tolist()):
    m = meta[idx]
    print(f"{val:.3f} {m['id']}: {m['caption'][:90]}")

gf_idxs = [j for j, m in enumerate(meta) if "Goldfinch" in m["id"]]
best_j = max(gf_idxs, key=lambda j: sims[j].item())
m = meta[best_j]
print(f"\nBest goldfinch: {sims[best_j].item():.3f} {m['id']}: {m['caption']}")
