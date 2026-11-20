"""Generate synthetic GLM routing cases with the upstream PyTorch router.

Usage: python generate_router_reference.py /path/to/modeling_glm4_moe_lite.py
The input must be Transformers v5.23.0's file with the SHA-256 below.
No checkpoint weights are used: an identity projection feeds explicit logits.
"""

import ast
import hashlib
import json
import pathlib
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

SOURCE_SHA256 = "9d267668ae8ba7f39fa52817be3524e44cbf39132244063218f6180e37f4066d"
source = pathlib.Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256, "unexpected upstream source"
tree = ast.parse(source)
router = next(node for node in tree.body if isinstance(node, ast.ClassDef)
              and node.name == "Glm4MoeLiteTopkRouter")
router.body[0].args.args[1].annotation = None
module = ast.Module(body=[router], type_ignores=[])
namespace = {"torch": torch, "nn": nn, "F": F}
exec(compile(module, "verified-upstream-router", "exec"), namespace)
Router = namespace["Glm4MoeLiteTopkRouter"]


def make_case(name, logits, bias, groups, selected_groups, top_k, normalize):
    logits = torch.tensor(logits, dtype=torch.float32)
    bias = torch.tensor(bias, dtype=torch.float32)
    count = logits.numel()
    raw = {"model_type": "glm4_moe_lite", "topk_method": "noaux_tc",
           "n_routed_experts": count, "n_group": groups, "topk_group": selected_groups,
           "num_experts_per_tok": top_k, "norm_topk_prob": normalize,
           "routed_scaling_factor": 1.8}
    config = SimpleNamespace(num_experts_per_tok=top_k, num_local_experts=count,
        hidden_size=count, routed_scaling_factor=1.8, n_group=groups,
        topk_group=selected_groups, norm_topk_prob=normalize)
    model = Router(config).eval()
    with torch.no_grad():
        model.weight.copy_(torch.eye(count))
        model.e_score_correction_bias.copy_(bias)
        _, weights, indices = model(logits.reshape(1, -1))
    pairs = sorted(zip(indices[0].tolist(), weights[0].tolist()))
    return {"name": name, "config": raw, "logits": logits.tolist(), "bias": bias.tolist(),
            "expected": [{"expert": expert, "weight": weight} for expert, weight in pairs]}


cases = [
    make_case("published_dimensions", [((i * 37) % 127 - 63) / 11 for i in range(64)],
              [((i * 11) % 23 - 11) / 100 for i in range(64)], 1, 1, 4, True),
    make_case("grouped_bias", [-2.2, -1.4, 0.8, 0.2, -0.1, 0.6, -1.1, 1.3],
              [0.5, 0.01, -0.2, 0.2, 0.1, 0.3, 0.6, -0.3], 2, 1, 2, True),
    make_case("top_one_normalized", [0.9, 0.1, -0.2, -1.0, -3.0, -4.0, -5.0, -6.0],
              [0.0] * 8, 2, 1, 1, True),
    make_case("top_one_unnormalized", [0.9, 0.1, -0.2, -1.0, -3.0, -4.0, -5.0, -6.0],
              [0.0] * 8, 2, 1, 1, False),
    make_case("negative_adjusted_scores", [-3.1, -2.2, -1.7, -2.9, -3.6, -3.2, -2.5, -3.4],
              [-2.0, -2.2, -2.1, -2.0, -3.0, -3.1, -2.9, -3.2], 2, 1, 2, True),
]
path = pathlib.Path(__file__).with_name("glm47-router-pytorch.json")
path.write_text(json.dumps({"source_sha256": SOURCE_SHA256, "cases": cases}, indent=2) + "\n")
