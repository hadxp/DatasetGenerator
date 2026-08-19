import os
import sys
import torch
from safetensors.torch import load_file, save_file

def load_convert_and_save_lora(lora_files: list[str], save_file_path: str, show_tensors: bool) -> None:
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
            if is_hunyuan:
                print("HunyuanVideo LoRA detected, converting to FramePack format")
                lora_sd = convert_hunyuan_to_framepack(lora_sd)
    
        # safetensors requires all tensors in the state dictionary to be contiguous in memory. If you modified the tensors or created slices of them, you may need to make them contiguous before saving
        state_dict = {k: v.clone().contiguous() for k, v in lora_sd.items()}
        if show_tensors:
            sd_keys = lora_sd.keys()
            print(str(sd_keys))
        save_file(state_dict, save_file_path)


def convert_hunyuan_to_framepack(
    lora_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Convert HunyuanVideo LoRA weights to FramePack format.
    """
    new_lora_sd = {}
    for key, weight in lora_sd.items():
        # if key.startswith("lora_unet_"):
        #     # hack? remove prefix from musubi tuner format
        #     key = key.replace("lora_unet_", "")
        if "double_blocks" in key:
            # print(f"Converting double_blocks HunyuanVideo LoRA key: {key}")
            key = key.replace("double_blocks", "transformer_blocks")
            key = key.replace("img_mod_linear", "norm1_linear")
            key = key.replace("img_attn_qkv", "attn_to_QKV")  # split later
            key = key.replace("img_attn_proj", "attn_to_out_0")
            key = key.replace("img_mlp_fc1", "ff_net_0_proj")
            key = key.replace("img_mlp_fc2", "ff_net_2")
            key = key.replace("txt_mod_linear", "norm1_context_linear")
            key = key.replace("txt_attn_qkv", "attn_add_QKV_proj")  # split later
            key = key.replace("txt_attn_proj", "attn_to_add_out")
            key = key.replace("txt_mlp_fc1", "ff_context_net_0_proj")
            key = key.replace("txt_mlp_fc2", "ff_context_net_2")
            # print(f"Converted double_blocks HunyuanVideo LoRA key: {key}")
        elif "single_blocks" in key:
            # print(f"Converting single_blocks HunyuanVideo LoRA key: {key}")
            key = key.replace("single_blocks", "single_transformer_blocks")
            key = key.replace("linear1", "attn_to_QKVM")  # split later
            key = key.replace("linear2", "proj_out")
            key = key.replace("modulation_linear", "norm_linear")
            # print(f"Converted single_blocks HunyuanVideo LoRA key: {key}")
        else:
            print(
                f"Unsupported module name: {key}, only double_blocks and single_blocks are supported"
            )
            continue

        if "QKVM" in key:
            # print(f"Converting QKVM HunyuanVideo LoRA key: {key}")
            # split QKVM into Q, K, V, M
            key_q = key.replace("QKVM", "q")
            key_k = key.replace("QKVM", "k")
            key_v = key.replace("QKVM", "v")
            key_m = key.replace("attn_to_QKVM", "proj_mlp")
            if "_down" in key or "alpha" in key:
                # copy QKVM weight or alpha to Q, K, V, M
                assert "alpha" in key or weight.size(1) == 3072, (
                    f"QKVM weight size mismatch: {key}. {weight.size()}"
                )
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
                new_lora_sd[key_m] = weight
            elif "_up" in key:
                # split QKVM weight into Q, K, V, M
                assert weight.size(0) == 21504, (
                    f"QKVM weight size mismatch: {key}. {weight.size()}"
                )
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 : 3072 * 3]
                new_lora_sd[key_m] = weight[3072 * 3 :]  # 21504 - 3072 * 3 = 12288
            else:
                print(f"Unsupported module name: {key}")
                continue
            # print(f"Converted QKVM HunyuanVideo LoRA key: {key}")
        elif "QKV" in key:
            # print(f"Converting QKV HunyuanVideo LoRA key: {key}")
            # split QKV into Q, K, V
            key_q = key.replace("QKV", "q")
            key_k = key.replace("QKV", "k")
            key_v = key.replace("QKV", "v")
            if "_down" in key or "alpha" in key:
                # copy QKV weight or alpha to Q, K, V
                assert "alpha" in key or weight.size(1) == 3072, (
                    f"QKV weight size mismatch: {key}. {weight.size()}"
                )
                new_lora_sd[key_q] = weight
                new_lora_sd[key_k] = weight
                new_lora_sd[key_v] = weight
            elif "_up" in key:
                # split QKV weight into Q, K, V
                assert weight.size(0) == 3072 * 3, (
                    f"QKV weight size mismatch: {key}. {weight.size()}"
                )
                new_lora_sd[key_q] = weight[:3072]
                new_lora_sd[key_k] = weight[3072 : 3072 * 2]
                new_lora_sd[key_v] = weight[3072 * 2 :]
            else:
                print(f"Unsupported module name: {key}")
                continue
            # print(f"Converted QKV HunyuanVideo LoRA key: {key}")
        else:
            # no split needed
            new_lora_sd[key] = weight

    return new_lora_sd


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
    show_tensors = args[3]
    load_convert_and_save_lora([lora_path], save_file_path, show_tensors=show_tensors)
    

if __name__ == "__main__":
    main()