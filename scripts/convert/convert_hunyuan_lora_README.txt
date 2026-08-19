use hunyuan lora in framepack lora training (diffusion-pipe):
1) convert hunyuan to musubi-tuner-framepack - datasetgenerator/scripts/convert_lora.py
2) convert musubi-tuner-framepack to diffusers-framepack format - musubi-tuner/convert_lora.py
3) convert diffusers-framepack to valid framepack-key-format - datasetgenerator/scripts/musubi_tuner_to_framepack.py