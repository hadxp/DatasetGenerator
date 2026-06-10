import requests
import os

# ================== CONFIG ==================
TB_URL = "http://192.168.0.30:6006"
TAGS = ["loss/current", "loss/epoch", "lr/unet"]
OUTPUT_DIR = "tensorboard_data"
RUN_NAME = "kdrkitten"
# ===========================================

all_data = {}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_runs():
    """Get list of available runs"""
    r = requests.get(f"{TB_URL}/data/plugin/scalars/tags")
    r.raise_for_status()
    return r.json()


def download_scalar(run: str, tag: str) -> str:
    """Download data for a specific run + tag"""
    url = f"{TB_URL}/data/plugin/scalars/scalars"
    params = {"run": run, "tag": tag, "format": "csv"}

    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.text


def main():
    runs_dict = get_runs()

    runs = list(runs_dict.keys())
    
    for run in runs:
        if RUN_NAME in run:
            print(f"Downloading data for run: **{run}**")
            for tag in TAGS:
                try:
                    content_json = download_scalar(run, tag)

                    all_data[f"{run}__{tag.replace('/', '_')}"] = {
                        'content': content_json
                    }
                except Exception as e:
                    print(f"   ❌ Failed {tag} → {e}")

    # Save consolidated CSV
    master_file_path = os.path.join(OUTPUT_DIR, f"{RUN_NAME}.csv")
    with open(master_file_path, "w") as master_file:
        for key, data in all_data.items():
            master_file.write(f"# {key}\n")
            master_file.write(data['content'])
            master_file.write("\n")
    
    print(f"\nAll done! Data saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()