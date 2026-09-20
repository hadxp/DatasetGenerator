import requests
import sys
import os

# ================== CONFIG ==================
TB_URL = "http://192.168.0.30:6006"
TAGS = {
    'diffusion-pipe': {
        'tags': [
            "train/loss",
            "train/epoch_loss",
            "train/ema_loss",
            "train/avr_loss",
            "train/avr_epoch_loss",
            "train/automagic_avg_lr",
            "train/lr",
        ],
        'texts': [
            "num train dataloader items/text_summary",
            "batch size/text_summary",
            "num repeats/text_summary",
            "total steps/text_summary",
            "num epochs/text_summary",
            "num train batches/text_summary",
            "data parallel world size/text_summary",
            "num train items/text_summary",
            "gradient accumulation steps/text_summary",
            "run_data/text_summary",
        ],
        'replace': ["train", "text_summary"],
    },

    'musubi-tuner': {
        'tags': [
            "loss/current",
            "loss/epoch",
            "lr/unet",
        ],
        'replace': ["loss", "unet"],
    },
}

def log_handler(tb_writer,
                log_type: str,
                log: dict[str, dict],
                x_axis: int,
                wandb_enable: bool):
    if not log:
        return
    match log_type:
        case 'scalar':
            for tag, val in log.items():
                tb_writer.add_scalar(tag=tag, scalar_value=val, global_step=x_axis)
                if wandb_enable:
                    import wandb
                    wandb.log(val, step=x_axis)
        case 'histogram':
            for tag, val in log.items():
                tb_writer.add_histogram(tag=tag, values=val, global_step=x_axis)
                if wandb_enable:
                    import wandb
                    wandb_hist_dict = {
                        tag: wandb.Histogram(val) for tag, val in log.items()
                    }
                    wandb.log(wandb_hist_dict, step=x_axis)
        case 'text':
            for tag, data in log.items():
                text = "\n".join(f"{k}: {v}" for k, v in data.items())
                #tb_writer.add_text(tag, text, global_step=0)
                i=0

logs = {
    "run_data": {
        "num train items": str(9),
        "num train batches": str(8),
        "num train dataloader items": str(7),
        "num repeats": str(6),
        "num epochs": str(5),
        "batch size": str(4),
        "gradient accumulation steps": str(3),
        "data parallel world size": str(2),
        "total steps": str(1),
    },
}

log_handler(None, "text", logs, 0, False)



OUTPUT_DIR = "tensorboard_data"
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


def download_text(run: str, tag: str, markdown: bool = False) -> str:
    """Download text events for a specific run + tag"""
    url = f"{TB_URL}/data/plugin/text/text"
    params = {"run": run, "tag": tag}
    params["markdown"] = str(markdown).lower() if markdown else "false"

    r = requests.get(url, params=params)
    r.raise_for_status()
    
    t = r.json()[0]['text']
    f = bare_text(t)
    
    return f


from html import unescape
import re

def bare_text(html: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", html)).strip()


def main():
    runs_dict: list[str] = get_runs()

    runs: list[str] = list(runs_dict.keys())
    
    RUN_NAME: str = sys.argv[1]
    RUN_NAME2: str = None
    try:
        RUN_NAME2 = sys.argv[2]
    except IndexError:
        pass
    
    for run in runs:
        if RUN_NAME in run:
            print(f"Downloading all data for run: **{run}**")
            for trainer in TAGS:
                for k in TAGS[trainer]:
                    if k == 'texts':
                        for text_tag in TAGS[trainer][k]:
                            try:
                                content_json = download_text(run, text_tag)

                                run_name = (
                                    run
                                    if run != "."
                                    else RUN_NAME2
                                    if RUN_NAME2 is not None
                                    else run
                                )

                                replace = TAGS[trainer]["replace"]
                                for k in replace:
                                    text_tag = text_tag.replace(k, "").replace("/", "")

                                all_data[f"{run_name}__{text_tag}"] = {
                                    "run": run_name,
                                    "content": content_json,
                                }
                            except Exception:
                                pass
                    elif k == 'tags':
                        for tag in TAGS[trainer][k]:
                            try:
                                content_json = download_scalar(run, tag)
            
                                run_name = run if run != '.' else RUN_NAME2 if RUN_NAME2 is not None else run
                                
                                replace = TAGS[trainer]['replace']
                                for k in replace:
                                    tag = tag.replace(k, '').replace('/', '')
                                sanitized_content = content_json.replace("Value", tag)
            
                                all_data[f"{run_name}__{tag}"] = {
                                    "run": run_name,
                                    "content": sanitized_content,
                                }
                            except Exception:
                                pass
                            

    # Save consolidated CSV
    master_file_path = os.path.join(OUTPUT_DIR, f"{RUN_NAME}.csv")
    with open(master_file_path, "w+") as master_file:
        master_file.write(f"{RUN_NAME2}:\n")
        for key, data in all_data.items():
            k = f"# {key}\n"
            v = data['content']
            
            master_file.write(k)
            master_file.write(v)
            master_file.write("\n")
    
    print(f"\nAll done! Data saved to '{master_file_path}'")


if __name__ == "__main__":
    main()