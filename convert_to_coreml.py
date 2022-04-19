import cv2
import torch
import coremltools as ct
import numpy as np


from UNet import UNet

MODEL_PATH = "checkpoints/model_causal.pt"

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

unet = UNet()
# unet = torch.load(MODEL_PATH, map_location=device).to(device)
unet.eval()
unet.exporting = True

dummy_input = torch.randn(1, 1, 128, 64).to(device)

import pdb
pdb.set_trace()
traced_model = torch.jit.trace(unet, (dummy_input))

ml_input = ct.TensorType(name="sample_input", shape=(1, 1, 128, 64))

# Convert to CoreML
model = ct.convert(
    traced_model,
    inputs=[ml_input],
)

model.save("speech_unet_causal.mlmodel")