import sys
from safetensors.torch import load_file, save_file

# Path to your existing float32 LoRA
input_path = sys.argv[1]
output_path = sys.argv[2]
dtype = sys.argv[3]

# Load the weights
state_dict = load_file(input_path)

# Cast to bfloat16
state_dict_bf16 = {k: v.to(dtype=dtype) for k, v in state_dict.items()}

# Save
save_file(state_dict_bf16, output_path)