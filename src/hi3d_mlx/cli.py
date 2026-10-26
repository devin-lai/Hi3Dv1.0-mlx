"""Command-line inference and selective checkpoint downloads."""
import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    from hi3d_mac.presets import PRESETS
    parser = argparse.ArgumentParser(description="Hi3D MLX: refine an image and coarse mesh on Apple silicon")
    sub = parser.add_subparsers(dest="command", required=True)
    infer = sub.add_parser("refine", help="Refine an existing coarse mesh")
    infer.add_argument("--image", type=Path, required=True)
    infer.add_argument("--coarse-mesh", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--preset", choices=tuple(PRESETS), default="fp16")
    infer.add_argument("--resolution", type=int, choices=(512, 1024, 1536), default=512)
    infer.add_argument("--seed", type=int, default=0)
    infer.add_argument("--steps", type=int)
    infer.add_argument("--attention", choices=("auto", "mlx", "flash"), default="auto")
    infer.add_argument("--model-dir", type=Path, help="Already assembled local model folder")
    infer.add_argument("--local-files-only", action="store_true")
    fetch = sub.add_parser("download", help="Fetch one weight variant and its dependencies")
    fetch.add_argument("--variant", choices=("fp16", "int8", "int4", "w8a8"), default="fp16")
    fetch.add_argument("--resolution", type=int, choices=(512, 1024, 1536), default=512)
    fetch.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "refine":
        if args.steps is not None and args.steps < 1:
            parser.error("--steps must be positive")
        if args.output.exists():
            parser.error(f"Output already exists: {args.output}; choose a new path")
        # The allocator setting must be present before Python initializes Metal.
        if argv is None and os.environ.get("MallocLargeCache") != "0":
            os.execve(sys.executable, [sys.executable, "-m", "hi3d_mlx", *sys.argv[1:]],
                      {**os.environ, "MallocLargeCache": "0", "MLX_ENABLE_TF32": "0"})
        from .api import refine
        values = vars(args).copy(); values.pop("command")
        refine(**values)
        print(f"Saved {args.output}")
    else:
        from .hub import prepare_models, prepare_encoders
        root = prepare_models(args.variant, args.resolution, local_files_only=args.local_files_only)
        prepare_encoders(local_files_only=args.local_files_only)
        print(f"Ready: {root}")
    return 0
