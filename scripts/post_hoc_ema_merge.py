import argparse
from pathlib import Path
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
        default=0.15,
        help="The value for sigma_rel",
    )
    parser.add_argument(
        "--limit",
        type=str,
        default=None,
    )
    return parser

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
    safetensor_files = []

    # Add files from main directory
    safetensor_files.extend(lora_main_dir_path.glob("*.safetensors"))

    # Add files from resume subdirectories
    for folder in lora_main_dir_path.iterdir():
        if folder.is_dir():
            safetensor_files.extend(folder.glob("*.safetensors"))

    # exclude all previous merges
    safetensor_files = [p for p in safetensor_files if "merge" not in p.stem]

    # exclude model.safetensorns (if present)
    safetensor_files = [p for p in safetensor_files if "model" not in p.stem]

    # exclude any orig safetenso files
    safetensor_files = [p for p in safetensor_files if "orig" not in p.stem]
    
    if no_const:
        safetensor_files = [p for p in safetensor_files
                            if "const" not in p.stem and "constant" not in p.stem]
    
    if limit:
        if "," in limit: # separator
            limitsplit = limit.split(",")
            files = []
            for epoch_limit in limitsplit:
                for file in safetensor_files:
                    if epoch_limit in file.stem:
                        files.append(file)
            safetensor_files = files
                        
    # Convert to strings
    files = [str(p) for p in safetensor_files]
    
    filess = [p.stem for p in safetensor_files]
    sorted_filess = sorted(filess, key=lambda p: p)
    #print(str(sorted_filess))
    
    merge_lora_weights_with_post_hoc_ema(files, False, 0.30, None, sigmarel, output_file)
    
    
if __name__ == "__main__":
    main()