import torch

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