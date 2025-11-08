#!/usr/bin/env python3
import argparse
import ast
import glob
import json
import os
import sys
from typing import Dict, List

from loop_utils.colmap_runner import ColmapRunner

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run COLMAP on a subset of images and export intrinsics/poses."
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Directory, glob pattern, image file, or txt file listing chunk images.",
    )
    parser.add_argument(
        "--workspace_root",
        default="./_colmap_chunks",
        help="Directory for intermediate COLMAP workspaces.",
    )
    parser.add_argument(
        "--chunk_name",
        default=None,
        help="Optional label for the chunk (defaults to chunk_<idx>).",
    )
    parser.add_argument(
        "--matcher",
        default="sequential",
        choices=["sequential", "exhaustive"],
        help="Matching strategy to use.",
    )
    parser.add_argument(
        "--sequential_overlap",
        type=int,
        default=5,
        help="Number of neighbors used by sequential matcher.",
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=2048,
        help="Max dimension for SIFT extraction (0 to disable).",
    )
    parser.add_argument(
        "--min_num_matches",
        type=int,
        default=15,
        help="Minimum matches before registering an image.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="Threads for COLMAP mapper (defaults to COLMAP internal setting).",
    )
    parser.add_argument(
        "--keep_workspace",
        action="store_true",
        help="Keep per-chunk COLMAP workspace on disk.",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU usage for SIFT/matching.",
    )
    parser.add_argument(
        "--camera_model",
        default="SIMPLE_RADIAL",
        help="Camera model used by the ImageReader.",
    )
    parser.add_argument(
        "--multi_camera",
        action="store_true",
        help="Do not assume a single camera for all images.",
    )
    parser.add_argument(
        "--colmap_binary",
        default="colmap",
        help="Name or path of the COLMAP executable.",
    )
    parser.add_argument(
        "--export_json",
        default=None,
        help="Optional path to write calibration metadata as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging from the runner.",
    )
    parser.add_argument(
        "--feature-option",
        dest="feature_option",
        action="append",
        default=[],
        help="Additional KEY=VALUE pair for feature_extractor (e.g. SiftExtraction.max_num_features=8000).",
    )
    parser.add_argument(
        "--matcher-option",
        dest="matcher_option",
        action="append",
        default=[],
        help="Additional KEY=VALUE pair for matcher (e.g. SequentialMatching.overlap=10).",
    )
    parser.add_argument(
        "--mapper-option",
        dest="mapper_option",
        action="append",
        default=[],
        help="Additional KEY=VALUE pair for mapper (e.g. Mapper.ba_refine_extra_params=1).",
    )
    return parser.parse_args()


def resolve_image_list(spec: str) -> List[str]:
    if os.path.isdir(spec):
        entries = sorted(os.listdir(spec))
        return [
            os.path.abspath(os.path.join(spec, entry))
            for entry in entries
            if entry.lower().endswith(IMAGE_EXTS)
        ]

    if os.path.isfile(spec):
        if spec.lower().endswith(".txt"):
            with open(spec, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            return [os.path.abspath(line) for line in lines]
        return [os.path.abspath(spec)]

    matches = glob.glob(spec)
    if matches:
        return sorted(
            [os.path.abspath(path) for path in matches if path.lower().endswith(IMAGE_EXTS)]
        )

    raise FileNotFoundError(f"Could not resolve any images from '{spec}'.")


def parse_option_pairs(pairs: List[str]) -> Dict[str, object]:
    options: Dict[str, object] = {}
    for raw in pairs or []:
        if "=" not in raw:
            raise ValueError(f"Option '{raw}' must be in KEY=VALUE format.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Option '{raw}' has an empty key.")
        value = value.strip()
        if not value:
            raise ValueError(f"Option '{raw}' has an empty value.")
        try:
            parsed_value = ast.literal_eval(value)
        except Exception:
            parsed_value = value
        options[key] = parsed_value
    return options


def main() -> None:
    args = parse_args()
    try:
        feature_args = parse_option_pairs(args.feature_option)
        matcher_args = parse_option_pairs(args.matcher_option)
        mapper_args = parse_option_pairs(args.mapper_option)
    except ValueError as exc:
        print(f"Error parsing COLMAP options: {exc}", file=sys.stderr)
        sys.exit(2)
    image_paths = resolve_image_list(args.images)
    if not image_paths:
        print(f"No images found for '{args.images}'.", file=sys.stderr)
        sys.exit(1)

    runner = ColmapRunner(
        workspace_root=args.workspace_root,
        colmap_binary=args.colmap_binary,
        matcher=args.matcher,
        sequential_overlap=args.sequential_overlap,
        use_gpu=not args.no_gpu,
        camera_model=args.camera_model,
        single_camera=not args.multi_camera,
        max_image_size=args.max_image_size if args.max_image_size > 0 else None,
        min_num_matches=args.min_num_matches,
        num_threads=args.num_threads,
        keep_workspaces=args.keep_workspace,
        verbose=args.verbose,
        feature_extractor_args=feature_args,
        matcher_args=matcher_args,
        mapper_args=mapper_args,
    )
    result = runner.run_for_chunk(image_paths, chunk_name=args.chunk_name)
    print(result.summary())
    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(result.to_serializable(), f, indent=2)
        print(f"Saved COLMAP metadata to {args.export_json}")


if __name__ == "__main__":
    main()
