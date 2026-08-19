import argparse
from pathlib import Path
from typing import List

from lora_post_hoc_ema import merge_lora_weights_with_post_hoc_ema


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "lora_dir",
        type=str,
        default=None,
        help="The name of the folder the dataset(s) to parse is in (comma separated)",
    )
    parser.add_argument(
        "--no_const", action="store_true", default=False, help="Excludes constant files"
    )
    parser.add_argument(
        "--sigmarel",
        type=float,
        default=0.25,
        help="The value for sigma_rel",
    )
    parser.add_argument(
        "--limit",
        type=str,
        default=None,
    )
    return parser


def _get_files_for_limit(limits: List[int], safetensor_files: List[Path]) -> List[Path]:
    files = []
    for epoch_limit in limits:
        for file in safetensor_files:
            epoch_str = str(epoch_limit)
            if ((epoch_str in file.stem) or (epoch_str in file.parent.stem)
                and (
                    ".safetensors" in file.suffix
                    or ".pkl" in file.suffix
                    or ".pt" in file.suffix
                )
            ):
                files.append(file)
    return files


def main():
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()

    lora_dir: str = args.lora_dir
    no_const: bool = args.no_const
    sigmarel: float = args.sigmarel
    limit: str = args.limit

    lora_main_dir_path = Path(lora_dir)
    output_file: str = str(lora_main_dir_path / "merge.safetensors")

    # Collect all safetensor files in a flat list
    safetensor_files: List[Path] = []

    # Add files from main directory
    safetensor_files.extend(lora_main_dir_path.glob("*.safetensors"))

    # Add files from resume subdirectories
    for folder in lora_main_dir_path.iterdir():
        if folder.is_dir():
            safetensor_files.extend(folder.glob("*.safetensors"))

    if limit:
        if "-" in limit:
            limits = limit.split("-")
            if len(limits) < 2:
                raise ValueError("At least two limit are required")
            limit_min = int(limits[0])  # min
            limit_max = int(limits[1])  # max
            limits = []
            # add 1 to limit_min until limit_max is reached
            while limit_min <= limit_max:
                limits.append(int(limit_min))
                limit_min += 1
            safetensor_files = _get_files_for_limit(limits, safetensor_files)

        if "," in limit:  # separator
            limits = limit.split(",")
            safetensor_files = _get_files_for_limit(limits, safetensor_files)

    # Convert to strings
    files = [str(p) for p in safetensor_files]

    # exclude all previous merges
    safetensor_files = [p for p in safetensor_files if "merge" not in p.stem]

    # exclude model.safetensorns (if present)
    safetensor_files = [p for p in safetensor_files if "model" not in p.stem]

    # exclude any orig safetensor files
    safetensor_files = [p for p in safetensor_files if "orig" not in p.stem]

    if no_const:
        safetensor_files = [
            p
            for p in safetensor_files
            if "const" not in p.stem and "constant" not in p.stem
        ]

    filess = [p.stem for p in safetensor_files]
    #sorted_filess = sorted(filess, key=lambda p: p)
    #print(str(sorted_filess))

    merge_lora_weights_with_post_hoc_ema(files, False, None, None, sigmarel, output_file)


if __name__ == "__main__":
    main()
