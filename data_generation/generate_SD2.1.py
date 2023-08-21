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
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to(device)

def fake_inpaint(img, comp_mask, device, vae):
    x = transforms.ToTensor()(img).to(device)[None]
    posterior = vae.encode(x).latent_dist
    z = posterior.mode()
    dec = vae.decode(z).sample[0]
    inpainted = transforms.ToPILImage()(dec.clamp_(0, 1))
    ret = Image.composite(inpainted, img, comp_mask)
    return ret


outputdir = "SD2.1"

image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

for i in range(0,67):
    row = captions.iloc[i]
    image = Image.open(row.image).resize((512,512),PIL.Image.Resampling.BILINEAR).convert("RGB")
    w, h = image.size
    caption = row.caption
    orig_image = copy.deepcopy(image)
    inputs = image_processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    arr_ = None
    for num in np.unique(result['segmentation'].numpy()):
        
        mask = (result['segmentation'].numpy() == num)
        visual_mask = (mask * 255).astype(np.uint8)

        if ((np.sum(visual_mask == 0) / (512*512)) < 0.7) and ((np.sum(visual_mask == 0) / (512*512)) > 0.2):
            arr_ = visual_mask
            break

    if type(arr_) == type(None):
        mask = (result['segmentation'].numpy() == 1)
        arr_ = (mask * 255).astype(np.uint8)

    mask = PIL.Image.fromarray(arr_).resize((512,512),PIL.Image.Resampling.BILINEAR)
    # mask = mask.point( lambda p: 255 if p >= 1 else 0 )
    blurred_mask = mask.filter(PIL.ImageFilter.GaussianBlur(radius=3))
    actual_mask = blurred_mask.point( lambda p: 255 if p >= 255 else 0 )
    black = Image.new("RGB", (512,512), 0)
    image = Image.composite(black, image, actual_mask)    


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
    original_image.save(os.path.join(outputdir, str(i)+".original.webp"), lossless=True)

model = YOLO("yolov8m.pt")

for i in range(67,100):
    row = captions.iloc[i]
    im = Image.open(row.image).resize((512,512),PIL.Image.Resampling.BILINEAR).convert("RGB")
    w, h = im.size
    caption = row.caption

    results = model.predict(im, conf=0.5)[0]
    
    if len(results.boxes.data) > 0:
        b = None
        for box in results.boxes.data:
            x, y, x_end, y_end, confidence, class_id = box
            if (((y_end - y) * (x_end - x)) / 512**2) < 0.7:
                break
        
    else:
        minsize=8*512/64
        x = round(random.random() * (w-minsize) + minsize/2)
        y = round(random.random() * (h-minsize) + minsize/2)
        cropw = round(random.random() * (w/2 - minsize)) + minsize
        croph = round(random.random() * (h/2 - minsize)) + minsize
        # print((x, y), (cropw, croph))
        x = max(0, round(x - cropw/2))
        y = max(0, round(y - croph/2))
        x_end = min(w, x + cropw)
        y_end = min(h, y + croph)

    mask = PIL.Image.new("L", im.size, 0)
    draw = PIL.ImageDraw.Draw(mask)
    draw.rectangle((x, y, x_end, y_end), fill=255)
    blurred_mask = mask.filter(PIL.ImageFilter.GaussianBlur(radius=3))
    actual_mask = blurred_mask.point( lambda p: 255 if p >= 255 else 0 )
    
    black = PIL.Image.new("RGB", im.size, 0)
    image = PIL.Image.composite(black, im, actual_mask) 

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