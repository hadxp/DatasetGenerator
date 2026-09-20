import torch


HIDDEN = 3072
QKV_OUT = HIDDEN * 3          # 9216
QKVM_OUT = 21504              # 3072*3 + 12288
MLP_OUT = QKVM_OUT - QKV_OUT  # 12288


def _is_down_or_alpha(key: str) -> bool:
    return any(s in key for s in ("lora_down", "lora_A", ".alpha", "_alpha"))


def _is_up(key: str) -> bool:
    return any(s in key for s in ("lora_up", "lora_B"))


def _swap_once(key: str, old: str, new: str) -> str:
    if old not in key:
        raise KeyError(f"{old!r} not in {key}")
    return key.replace(old, new, 1)


def _rename_non_attn(key: str) -> str | None:
    if "single_transformer_blocks" in key:
        key = key.replace("single_transformer_blocks", "single_blocks")
        key = key.replace("proj_out", "linear2")
        key = key.replace("norm_linear", "modulation_linear")
        return key
    if "transformer_blocks" in key:
        key = key.replace("transformer_blocks", "double_blocks")
        key = key.replace("norm1_context_linear", "txt_mod_linear")
        key = key.replace("norm1_linear", "img_mod_linear")
        key = key.replace("attn_to_out_0", "img_attn_proj")
        key = key.replace("ff_net_0_proj", "img_mlp_fc1")
        key = key.replace("ff_net_2", "img_mlp_fc2")
        key = key.replace("attn_to_add_out", "txt_attn_proj")
        key = key.replace("ff_context_net_0_proj", "txt_mlp_fc1")
        key = key.replace("ff_context_net_2", "txt_mlp_fc2")
        return key
    return None


def _fuse_down_or_alpha(weights: list[torch.Tensor]) -> torch.Tensor:
    """Hunyuan fused layers have one A/alpha. Prefer q's tensor; assert they match if possible."""
    ref = weights[0]
    for w in weights[1:]:
        if w.shape != ref.shape or not torch.equal(w, ref):
            # Native FramePack LoRA: A_q/A_k/A_v differ. Best-effort = use q's A.
            # Exact fused LoRA is impossible in that case.
            return ref
    return ref


def _collect(lora_sd, key_q, parts):
    keys = [_swap_once(key_q, parts[0], p) for p in parts]
    if not all(k in lora_sd for k in keys):
        return None
    return keys


def convert_framepack_to_hunyuan(
    lora_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Inverse of convert_hunyuan_to_framepack.

    Exact for LoRAs that were split from Hunyuan (shared lora_down / alpha).
    Best-effort for LoRAs trained natively on FramePack (separate q/k/v downs).
    Only double/single stream blocks are converted, same as the forward function.
    """
    new_lora_sd: dict[str, torch.Tensor] = {}
    processed: set[str] = set()

    for key, weight in lora_sd.items():
        if key in processed:
            continue

        # --- single stream: attn_to_q/k/v + proj_mlp  ->  linear1 (QKVM) ---
        if "single_transformer_blocks" in key and "attn_to_q" in key:
            parts = ("attn_to_q", "attn_to_k", "attn_to_v", "proj_mlp")
            keys = _collect(lora_sd, key, parts)
            if keys is None:
                print(f"Incomplete QKVM group for {key}, skipping")
                continue
            processed.update(keys)
            hunyuan_key = (
                key.replace("single_transformer_blocks", "single_blocks")
                   .replace("attn_to_q", "linear1")
            )
            if _is_down_or_alpha(key):
                new_lora_sd[hunyuan_key] = _fuse_down_or_alpha([lora_sd[k] for k in keys])
            elif _is_up(key):
                wq, wk, wv, wm = (lora_sd[k] for k in keys)
                assert wq.size(0) == HIDDEN and wk.size(0) == HIDDEN and wv.size(0) == HIDDEN, (
                    f"QKVM up slice mismatch: {[lora_sd[k].shape for k in keys]}"
                )
                assert wm.size(0) == MLP_OUT, f"proj_mlp up size {wm.shape}, expected first dim {MLP_OUT}"
                new_lora_sd[hunyuan_key] = torch.cat([wq, wk, wv, wm], dim=0)
            else:
                print(f"Unsupported QKVM suffix: {key}")
            continue

        # --- double stream image: attn_to_q/k/v  ->  img_attn_qkv ---
        if (
            "transformer_blocks" in key
            and "single_transformer_blocks" not in key
            and "attn_to_q" in key
        ):
            parts = ("attn_to_q", "attn_to_k", "attn_to_v")
            keys = _collect(lora_sd, key, parts)
            if keys is None:
                print(f"Incomplete image QKV group for {key}, skipping")
                continue
            processed.update(keys)
            hunyuan_key = (
                key.replace("transformer_blocks", "double_blocks")
                   .replace("attn_to_q", "img_attn_qkv")
            )
            if _is_down_or_alpha(key):
                new_lora_sd[hunyuan_key] = _fuse_down_or_alpha([lora_sd[k] for k in keys])
            elif _is_up(key):
                wq, wk, wv = (lora_sd[k] for k in keys)
                assert all(w.size(0) == HIDDEN for w in (wq, wk, wv)), (
                    f"QKV up slice mismatch: {[lora_sd[k].shape for k in keys]}"
                )
                new_lora_sd[hunyuan_key] = torch.cat([wq, wk, wv], dim=0)
            else:
                print(f"Unsupported QKV suffix: {key}")
            continue

        # --- double stream text: attn_add_q/k/v_proj  ->  txt_attn_qkv ---
        if "transformer_blocks" in key and "attn_add_q_proj" in key:
            parts = ("attn_add_q_proj", "attn_add_k_proj", "attn_add_v_proj")
            keys = _collect(lora_sd, key, parts)
            if keys is None:
                print(f"Incomplete text QKV group for {key}, skipping")
                continue
            processed.update(keys)
            hunyuan_key = (
                key.replace("transformer_blocks", "double_blocks")
                   .replace("attn_add_q_proj", "txt_attn_qkv")
            )
            if _is_down_or_alpha(key):
                new_lora_sd[hunyuan_key] = _fuse_down_or_alpha([lora_sd[k] for k in keys])
            elif _is_up(key):
                wq, wk, wv = (lora_sd[k] for k in keys)
                assert all(w.size(0) == HIDDEN for w in (wq, wk, wv)), (
                    f"text QKV up slice mismatch: {[lora_sd[k].shape for k in keys]}"
                )
                new_lora_sd[hunyuan_key] = torch.cat([wq, wk, wv], dim=0)
            else:
                print(f"Unsupported text QKV suffix: {key}")
            continue

        # leftover members of a fused group are consumed above
        if any(
            s in key
            for s in (
                "attn_to_k",
                "attn_to_v",
                "proj_mlp",
                "attn_add_k_proj",
                "attn_add_v_proj",
            )
        ):
            continue

        new_key = _rename_non_attn(key)
        if new_key is None:
            print(f"Unsupported module name: {key}")
            continue
        new_lora_sd[new_key] = weight

    return new_lora_sd