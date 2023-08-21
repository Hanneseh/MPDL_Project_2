import os
import json
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import PIL
from tqdm.auto import tqdm
from ultralytics import YOLO
import copy
from diffusers import StableDiffusionInpaintPipeline
import torch
from torchvision import transforms
from skimage import feature
import torchvision.transforms as T
import io
import cv2
from scipy import ndimage

BASE_PATH = 'train2017/'
with open(f'captions_train2017.json', 'r') as f:
    data = json.load(f)
    data = data['annotations']

img_cap_pairs = []

for sample in data:
    img_name = '%012d.jpg' % sample['image_id']
    img_cap_pairs.append([img_name, sample['caption']])

captions = pd.DataFrame(img_cap_pairs, columns=['image', 'caption'])
captions['image'] = captions['image'].apply(
    lambda x: f'{BASE_PATH}train2017/{x}'
)
captions = captions.sort_values(by=["image"])
captions = captions.drop_duplicates(subset=['image'], keep='first')
captions = captions[:20000]
captions = captions.reset_index(drop=True)


device = torch.device("cuda", 2)
    
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float32
).to(device)

def fake_inpaint(img, comp_mask, device, vae):
    x = transforms.ToTensor()(img).to(device)[None]
    posterior = vae.encode(x).latent_dist
    z = posterior.mode()
    dec = vae.decode(z).sample[0]
    inpainted = transforms.ToPILImage()(dec.clamp_(0, 1))
    ret = Image.composite(inpainted, img, comp_mask)
    return ret


outputdir = "data_qmd/"
for i in range(7500,10000):
    row = captions.iloc[i]
    image = Image.open(row.image).resize((512,512),PIL.Image.Resampling.BILINEAR).convert('RGB')
    orig_image = copy.deepcopy(image)
    w, h = image.size
    caption = row.caption

    mask = PIL.Image.open(f"qd_imd/train/0{i}_train.png").convert("L").resize((512,512),PIL.Image.Resampling.BILINEAR)
    mask = PIL.ImageOps.invert(mask)

    mask = ndimage.maximum_filter(mask, size=15)
    mask = PIL.Image.fromarray(mask).resize((512,512),PIL.Image.Resampling.BILINEAR)
    blurred_mask = mask.filter(PIL.ImageFilter.GaussianBlur(radius=3))
    actual_mask = blurred_mask.point( lambda p: 255 if p >= 255 else 0 )

    black = Image.new("RGB", image.size, 0)
    image = Image.composite(black, orig_image, actual_mask)    

    masked_image, mask, comp_mask, original_image = image, actual_mask, blurred_mask, orig_image

    # inpaint
    inpainted = pipe(prompt=caption, image=masked_image, mask_image=mask).images[0]
    inpainted.save(os.path.join(outputdir, str(i)+".inpainted.webp"), lossless=True)
    inpainted = PIL.Image.composite(inpainted, masked_image, comp_mask)
    
    #fake inpaint
    fakeinpainted = fake_inpaint(original_image, comp_mask, device=device, vae=pipe.vae)
    
    inpainted.save(os.path.join(outputdir, str(i)+".realfake.webp"), lossless=True)
    fakeinpainted.save(os.path.join(outputdir, str(i)+".fakefake.webp"), lossless=True)
    comp_mask.save(os.path.join(outputdir, str(i)+".mask.webp"), lossless=True)
    orig_image.save(os.path.join(outputdir, str(i)+".original.webp"), lossless=True)
