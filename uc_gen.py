#=====================================
#    Unconditional Generation 
#===================================

#!pip install -q diffusers
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_id = 'Docty/ddpm_ae_photos'

# load model and scheduler
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

#pipe.enable_attention_slicing()

# run pipeline in inference (sample random noise and denoise)
image = pipe(num_inference_steps=100)


image['images'][0]
