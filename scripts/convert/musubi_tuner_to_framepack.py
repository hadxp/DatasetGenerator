import re, safetensors.torch

IN = 'v1ctor1a_01_epoch70_framepack_diffusers.safetensors'
OUT = 'v1ctor1a_01_epoch70_framepack_diffusers_2.safetensors'

# tokens whose internal underscore must be preserved (checked longest-first)
PROTECTED = sorted([
    'single_transformer_blocks', 'transformer_blocks',
    'context_embedder', 'x_embedder', 'time_text_embed',
    'timestep_embedder', 'text_embedder', 'guidance_embedder',
    'ff_context', 'norm1_context', 'norm_out', 'proj_mlp', 'proj_out',
    'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out',
    'norm_added_q', 'norm_added_k', 'to_out', 'to_q', 'to_k', 'to_v',
    'linear_1', 'linear_2', 'pos_embed',
], key=len, reverse=True)

def to_dotted(base):
    for tok in PROTECTED:
        base = base.replace(tok, tok.replace('_', chr(0)))
    base = base.replace('_', '.').replace(chr(0), '_')
    return base

sd = safetensors.torch.load_file(IN)
out = {}
for k, v in sd.items():
    if k.endswith('.lora_down.weight'):
        base = re.sub(r'^lora_unet_', '', k[:-len('.lora_down.weight')])
        new_key = f'{to_dotted(base)}.lora_A.default.weight'
    elif k.endswith('.lora_up.weight'):
        base = re.sub(r'^lora_unet_', '', k[:-len('.lora_up.weight')])
        new_key = f'{to_dotted(base)}.lora_B.default.weight'
    else:
        continue
    out[new_key] = v

safetensors.torch.save_file(out, OUT)
