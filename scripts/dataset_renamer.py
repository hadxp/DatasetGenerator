import sys
import argparse
from pathlib import Path
from utils import get_image_files, get_video_files

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "dataset",
        type=str,
        default=None,
        help="The name of the folder the dataset(s) to parse is in (comma separated)",
    )
    parser.add_argument(
        "--search_dir",
        type=str,
        default=None,
        help="The folder to search the datasets in",
    )
    return parser

def main():
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    dataset_names_arg:str = args.dataset
    search_dir: str = args.search_dir
    
    if dataset_names_arg is None:
        print("No dataset(s) specified, cannot continue")
        sys.exit(1)

    if search_dir is None:
        datasets_path = Path.cwd() / "datasets"
    else:
        datasets_path = Path(search_dir)

    dataset_names_arg_arr = args.dataset.split(",")

    print(f"Using dataset search dir: {datasets_path}")

    for dataset_name in dataset_names_arg_arr:
        if not datasets_path.exists():
            datasets_path.mkdir()
            sys.exit(0)

        dataset_name_folder: Path = None

        for folder in datasets_path.iterdir():
            if folder.is_dir() and dataset_name in folder.name:
                dataset_name_folder = folder

        if dataset_name_folder is None:
            raise Exception(f"No dataset named '{dataset_name}' found in directory '{datasets_path}'")

        dataset_dir = datasets_path / dataset_name_folder

        print(f"Dataset folder is: {dataset_name_folder}")

        source_dir = dataset_dir / "dataset"  # set the source image directory path

        if not source_dir.exists():
            print(f"Error: Source dataset directory '{source_dir}' does not exist.")
            sys.exit(1)

        # Get image files
        print(f"Scanning for files in {source_dir}...")
        files = get_image_files(source_dir)
        if len(files) <= 0:
            files = get_video_files(source_dir)

        if not files:
            print("No image or video files found in the source directory.")
            sys.exit(1)
            
        for i, file in enumerate(files, start=1):
            new_file_name = f"{i}{file.suffix}"
            file.rename(file.parent / new_file_name)

if __name__ == "__main__":
    main()
