# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2026 8796n <info@8796.jp>
"""Export Depth Anything V2 Small to ONNX for the OBS native filter (obs-near-real3d).

Produces the depth model the plugin loads at runtime
(models/depth_anything_v2_small.onnx). Default output is **FP16** (half-precision
weights, FP32 input/output via cast wrappers so it stays a drop-in for the
plugin). On the RTX 3060 / DirectML this is ~1.85x faster than FP32 (10.8 vs
20.0 ms at 392) with negligible quality loss (max|Δ| vs FP32 ~0.012). FP16 export
needs CUDA; with --no-fp16 (or no CUDA) it falls back to an FP32 graph.

The input shape is fixed (no dynamic axes -- DirectML is ~5x slower with them).
DA V2 (DINOv2 ViT, patch 14) accepts any width/height that are multiples of 14,
so a non-square (e.g. 16:9) input can be exported to match the source aspect.

  python tools/export_onnx.py                            # 392x392, FP16 (currently shipped)
  python tools/export_onnx.py --width 448 --height 252   # 16:9, FP16 (both multiples of 14)
  python tools/export_onnx.py --no-fp16                  # FP32 graph

Requires: torch, transformers, onnx, onnxruntime (+ CUDA for FP16).
Note: a non-square model also needs the plugin's inference path generalised from
the square INFER_SIZE to INFER_W x INFER_H before it can be used.
"""
from __future__ import annotations

import argparse
import os

import onnx
import torch
from onnx import TensorProto, helper
from transformers import AutoModelForDepthEstimation

# repo root = parent of this tools/ dir, so --out resolves regardless of CWD
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "models", "depth_anything_v2_small.onnx")


class DepthWrapper(torch.nn.Module):
    """ONNX output is a plain (N,1,H,W) depth tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values).predicted_depth
        return out.unsqueeze(1) if out.dim() == 3 else out


def wrap_fp32_io(src, dst):
    """Wrap an FP16-I/O model so its graph I/O is FP32 (cast nodes), keeping the
    network internally FP16. Lets the plugin keep feeding/reading FP32."""
    m = onnx.load(src)
    g = m.graph
    inp, outp = g.input[0].name, g.output[0].name
    ih, oh = inp + "_h", outp + "_h"
    for nd in g.node:
        nd.input[:] = [ih if x == inp else x for x in nd.input]
        nd.output[:] = [oh if x == outp else x for x in nd.output]
    g.node.insert(0, helper.make_node("Cast", [inp], [ih],
                                      to=TensorProto.FLOAT16, name="in_cast"))
    g.node.append(helper.make_node("Cast", [oh], [outp],
                                   to=TensorProto.FLOAT, name="out_cast"))
    g.input[0].type.tensor_type.elem_type = TensorProto.FLOAT
    g.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
    onnx.save(m, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--width", type=int, default=392, help="input width, multiple of 14")
    ap.add_argument("--height", type=int, default=392, help="input height, multiple of 14")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()
    assert args.width % 14 == 0 and args.height % 14 == 0, \
        "width and height must be multiples of 14"

    fp16 = not args.no_fp16 and torch.cuda.is_available()
    if not args.no_fp16 and not fp16:
        print("[warn] CUDA not available -> exporting FP32 instead of FP16")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"loading {args.repo} ...")
    wrapper = DepthWrapper(
        AutoModelForDepthEstimation.from_pretrained(args.repo).eval()).eval()

    # NCHW: torch tensors are (N, C, H, W)
    if fp16:
        wrapper = wrapper.half().cuda()
        dummy = torch.randn(1, 3, args.height, args.width,
                            dtype=torch.float16, device="cuda")
        tmp = args.out + ".f16.tmp"
        print(f"exporting FP16 ({args.width}x{args.height}) ...")
        torch.onnx.export(wrapper, (dummy,), tmp, input_names=["pixel_values"],
                          output_names=["depth"], opset_version=17,
                          do_constant_folding=True, dynamo=False)
        wrap_fp32_io(tmp, args.out)   # FP32 I/O wrapper, FP16 internals
        os.remove(tmp)
    else:
        dummy = torch.randn(1, 3, args.height, args.width)
        print(f"exporting FP32 ({args.width}x{args.height}) ...")
        torch.onnx.export(wrapper, (dummy,), args.out, input_names=["pixel_values"],
                          output_names=["depth"], opset_version=17,
                          do_constant_folding=True, dynamo=False)

    sz = os.path.getsize(args.out)
    print(f"done: {args.out} ({sz/1e6:.1f} MB, {'FP16' if fp16 else 'FP32'})")

    try:  # sanity: FP32-in/out, correct shape
        import numpy as np
        import onnxruntime as ort
        s = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        x = np.random.randn(1, 3, args.height, args.width).astype(np.float32)
        got = s.run(["depth"], {"pixel_values": x})[0]
        print(f"onnxruntime check: out {got.shape} {got.dtype}")
    except Exception as e:  # noqa: BLE001
        print(f"[skip ort check] {e}")


if __name__ == "__main__":
    main()
