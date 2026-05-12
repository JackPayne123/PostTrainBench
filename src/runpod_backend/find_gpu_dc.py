"""Find datacenter(s) with stock for a given GPU type + CUDA constraint.

Why this script exists:
    On 2026-05-12 a manual sweep with a hardcoded DC list (~9 DCs) reported
    "only H100 CA-MTL-1 has A100/CUDA-12.8+" — wrong. RunPod actually has
    46 datacenters and US-MO-1 had A100 SXM 80GB at $1.39/hr. The hardcoded
    list missed US-MO-1, US-NC-1/2, US-TX-{1..6}, US-PA-1, EUR-IS-{1..4},
    EUR-NO-{1..2}, AP-{IN,JP}-1, SEA-SG-1, CA-MTL-{2,3,4}, etc.

    Two foot-guns the manual sweep hit:
      1. Hardcoded DC list — drifts behind the actual DC catalog.
      2. `lowestPrice(input:{... dataCenterId: X})` returns `null` (not an
         error) when stock=0 in that DC. Silent miss if the iteration list
         is wrong.

    This script queries `dataCenters` first to get the live DC list, then
    iterates against it. Pass `--cuda` to filter by minimum driver version.

Usage:
    PYTHONPATH=. python src/runpod_backend/find_gpu_dc.py \\
        --gpu "NVIDIA A100-SXM4-80GB" --cuda 12.8,12.9
    PYTHONPATH=. python src/runpod_backend/find_gpu_dc.py --gpu-fallback \\
        --cuda 12.8,12.9
    # `--gpu-fallback` walks a curated price-ordered list of compatible
    # GPUs for 9B+ workloads (A100/H100/H200), exits at first available.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNPOD_API_URL = "https://api.runpod.io/graphql"

# 9B+ workload fallback order (cheapest first, all CUDA-12.8+ compatible).
# Source: RunPod's GPU listing 2026-05-12. Add cards here as new SKUs land.
GPU_FALLBACK_ORDER = [
    "NVIDIA A100 80GB PCIe",      # $1.19/hr — when available
    "NVIDIA A100-SXM4-80GB",      # $1.39/hr
    "NVIDIA H100 NVL",            # $2.59/hr
    "NVIDIA H100 80GB HBM3",      # $2.69/hr
    "NVIDIA H100 PCIe",           # $2.99/hr
    "NVIDIA H200 SXM",            # higher; only if absolutely needed
]


def gql(query: str, variables: dict | None = None) -> dict:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("RUNPOD_API_KEY missing; source .env first")
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = request.Request(
        RUNPOD_API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Cloudflare in front of RunPod's GraphQL blocks default urllib UA
            # with 403 (error 1010). Any non-default UA bypasses the block.
            "User-Agent": "ptb-find-gpu-dc/1.0",
        },
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            if "errors" in data:
                sys.exit(f"GraphQL error: {data['errors']}")
            return data["data"]
    except (HTTPError, URLError) as e:
        sys.exit(f"HTTP error: {e}")


def list_datacenters() -> list[str]:
    """Live list — do NOT hardcode."""
    d = gql("query { dataCenters { id } }")
    return [dc["id"] for dc in d["dataCenters"]]


def lowest_price(gpu_id: str, dc_id: str, cuda_versions: list[str]) -> dict | None:
    """Returns {'price': float, 'stockStatus': str} or None if unavailable."""
    filt_parts = ["gpuCount: 1", f'dataCenterId: "{dc_id}"']
    if cuda_versions:
        cv = ",".join(f'"{v}"' for v in cuda_versions)
        filt_parts.append(f"allowedCudaVersions: [{cv}]")
    filt = ", ".join(filt_parts)
    q = (
        "query { gpuTypes(input:{id:\"" + gpu_id + "\"}) { lowestPrice(input:{"
        + filt + "}) { uninterruptablePrice stockStatus } } }"
    )
    d = gql(q)
    g = (d.get("gpuTypes") or [None])[0]
    lp = (g or {}).get("lowestPrice") if g else None
    if not lp or lp.get("uninterruptablePrice") is None:
        return None
    return {"price": lp["uninterruptablePrice"], "stockStatus": lp.get("stockStatus")}


def sweep(gpu_id: str, cuda_versions: list[str]) -> list[tuple[str, dict]]:
    """Return list of (dc_id, lowest_price_info) for available DCs, ordered by price."""
    dcs = list_datacenters()
    hits: list[tuple[str, dict]] = []
    for dc in dcs:
        info = lowest_price(gpu_id, dc, cuda_versions)
        if info:
            hits.append((dc, info))
    hits.sort(key=lambda x: x[1]["price"])
    return hits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gpu",
        help="GPU type id (e.g. 'NVIDIA A100-SXM4-80GB'). Use exact `gpuTypes.id`.",
    )
    p.add_argument(
        "--gpu-fallback",
        action="store_true",
        help="Walk GPU_FALLBACK_ORDER cheapest-first; exit at first hit.",
    )
    p.add_argument(
        "--cuda",
        default="12.8,12.9",
        help="Comma-separated allowed CUDA versions (driver-side). "
             "Default 12.8,12.9 matches our :27 image (vllm 0.20.2 + torch 2.11).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cuda_versions = [v.strip() for v in args.cuda.split(",") if v.strip()]
    if not args.gpu and not args.gpu_fallback:
        sys.exit("Pass --gpu <id> or --gpu-fallback")

    if args.gpu_fallback:
        for gpu in GPU_FALLBACK_ORDER:
            hits = sweep(gpu, cuda_versions)
            if hits:
                print(f"{gpu}: {len(hits)} DC(s) available")
                for dc, info in hits:
                    print(f"  ${info['price']:.2f}/hr  stock={info['stockStatus']}  {dc}")
                print(f"\nRecommended: RUNPOD_GPU_TYPE_ID=\"{gpu}\" RUNPOD_DATACENTER_ID={hits[0][0]}")
                return
            print(f"{gpu}: no DC available with CUDA {cuda_versions}")
        sys.exit("\nNo compatible GPU found in any DC. Try widening --cuda or waiting.")
    else:
        hits = sweep(args.gpu, cuda_versions)
        if not hits:
            sys.exit(f"{args.gpu}: no DC has stock with CUDA {cuda_versions}")
        for dc, info in hits:
            print(f"  ${info['price']:.2f}/hr  stock={info['stockStatus']}  {dc}")
        cheapest_dc = hits[0][0]
        print(f"\nRecommended: RUNPOD_GPU_TYPE_ID=\"{args.gpu}\" RUNPOD_DATACENTER_ID={cheapest_dc}")


if __name__ == "__main__":
    main()
