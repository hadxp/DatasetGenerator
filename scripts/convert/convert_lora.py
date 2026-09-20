import os
import sys
import torch
from safetensors.torch import load_file, save_file
from convert_hunyuan_to_framepack import convert_hunyuan_to_framepack
from convert_framepack_to_hunyuan import convert_framepack_to_hunyuan

def load_convert_and_save_lora(lora_files: list[str], save_file_path: str, mode: str, show_tensors: bool) -> None:
    for lora_file in lora_files:
        # Load LoRA safetensors file
        lora_sd = load_file(lora_file)

        # Check the format of the LoRA file
        keys = list(lora_sd.keys())
        if keys[0].startswith("lora_unet_"):
            print("Musubi Tuner LoRA detected")
        else:
            transformer_prefixes = [
                "diffusion_model",
                "transformer",
            ]  # to ignore Text Encoder modules
            lora_suffix = None
            prefix = None
            for key in keys:
                if lora_suffix is None and "lora_A" in key:
                    lora_suffix = "lora_A"
                if prefix is None:
                    pfx = key.split(".")[0]
                    if pfx in transformer_prefixes:
                        prefix = pfx
                if lora_suffix is not None and prefix is not None:
                    break

            if lora_suffix == "lora_A" and prefix is not None:
                # convert to musubi-tuner lora style (layout)
                print("Diffusion-pipe (?) LoRA detected, converting to FramePack format")
                lora_sd = convert_from_diffusion_pipe_or_something(
                    lora_sd, "lora_unet_"
                )

            else:
                print(f"LoRA file format not recognized: {os.path.basename(lora_file)}")
                lora_sd = None

        if lora_sd is not None:
            # Check LoRA is for FramePack or for HunyuanVideo
            is_hunyuan = False
            for key in lora_sd.keys():
                if "double_blocks" in key or "single_blocks" in key:
                    is_hunyuan = True
                    break
            if is_hunyuan and mode == "framepack":
                print("HunyuanVideo LoRA detected, converting to FramePack format")
                lora_sd = convert_hunyuan_to_framepack(lora_sd)
            elif is_hunyuan and mode == "hunyuan":
                print("HunyuanVideo LoRA detected -> no conversion")
                return
            elif not is_hunyuan and mode == "hunyuan":
                print("FramePack LoRA detected, converting to HunyuanVideo format")
                lora_sd = convert_framepack_to_hunyuan(lora_sd)
            elif not is_hunyuan and mode == "framepack":
                print("FramePack LoRA detected -> no conversion")
                return
    
        # safetensors requires all tensors in the state dictionary to be contiguous in memory. If you modified the tensors or created slices of them, you may need to make them contiguous before saving
        state_dict = {k: v.clone().contiguous() for k, v in lora_sd.items()}
        if show_tensors:
            sd_keys = lora_sd.keys()
            print(str(sd_keys))
        save_file(state_dict, save_file_path)


def convert_from_diffusion_pipe_or_something(
    lora_sd: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """
    Convert LoRA weights to the format used by the diffusion pipeline to Musubi Tuner.
    Copy from Musubi Tuner repo.
    """
    # convert from diffusers(?) to default LoRA
    # Diffusers format: {"diffusion_model.module.name.lora_A.weight": weight, "diffusion_model.module.name.lora_B.weight": weight, ...}
    # default LoRA format: {"prefix_module_name.lora_down.weight": weight, "prefix_module_name.lora_up.weight": weight, ...}

    # note: Diffusers has no alpha, so alpha is set to rank
    new_weights_sd = {}
    lora_dims = {}
    for key, weight in lora_sd.items():
        diffusers_prefix, key_body = key.split(".", 1)
        if diffusers_prefix != "diffusion_model" and diffusers_prefix != "transformer":
            print(f"unexpected key: {key} in diffusers format")
            continue

        new_key = (
            f"{prefix}{key_body}".replace(".", "_")
            .replace("_lora_A_", ".lora_down.")
            .replace("_lora_B_", ".lora_up.")
        )
        new_weights_sd[new_key] = weight

        lora_name = new_key.split(".")[0]  # before first dot
        if lora_name not in lora_dims and "lora_down" in new_key:
            lora_dims[lora_name] = weight.shape[0]

    # add alpha with rank
    for lora_name, dim in lora_dims.items():
        new_weights_sd[f"{lora_name}.alpha"] = torch.tensor(dim)

    return new_weights_sd

def main():
    args = sys.argv
    lora_path = args[1]
    save_file_path = args[2]
    mode = args[3]
    show_tensors = args[4]
    load_convert_and_save_lora([lora_path], save_file_path, mode, show_tensors=show_tensors)
    

if __name__ == "__main__":
    main()