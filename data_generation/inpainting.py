from diffusers import StableDiffusionInpaintPipeline
import torch
import os
import glob
import tqdm
import re
import json
import PIL
import random
import copy
from torchvision import transforms
import fire


def generate_random_mask(image:PIL.Image.Image, minsize=8*512/64, maskradius=3):
    w, h = image.size
    orig_image = copy.deepcopy(image)
    # select random top left corner
    x = round(random.random() * (w-minsize) + minsize/2)
    y = round(random.random() * (h-minsize) + minsize/2)
    cropw = round(random.random() * (w/2 - minsize)) + minsize
    croph = round(random.random() * (h/2 - minsize)) + minsize
    # print((x, y), (cropw, croph))
    x = max(0, round(x - cropw/2))
    y = max(0, round(y - croph/2))
    x_end = min(w, x + cropw)
    y_end = min(h, y + croph)
    mask = PIL.Image.new("L", image.size, 0)
    draw = PIL.ImageDraw.Draw(mask)
    draw.rectangle((x, y, x_end, y_end), fill=255)
    blurred_mask = mask.filter(PIL.ImageFilter.GaussianBlur(radius=maskradius))
    actual_mask = blurred_mask.point( lambda p: 255 if p >= 255 else 0 )
    # print((w, h), (x, y), (x_end, y_end))
    black = PIL.Image.new("RGB", image.size, 0)
    image = PIL.Image.composite(black, orig_image, actual_mask)    
    return image, actual_mask, blurred_mask, orig_image


def fake_inpaint(img, comp_mask, device, vae):
    x = transforms.ToTensor()(img).to(device)[None]
    posterior = vae.encode(x).latent_dist
    z = posterior.mode()
    dec = vae.decode(z).sample[0]
    inpainted = transforms.ToPILImage()(dec.clamp_(0, 1))
    ret = PIL.Image.composite(inpainted, img, comp_mask)
    return ret


def main(inpath:str="../datasets/laion_real/laion-aestheticsv2-6.5plus_splits_10000_0.9_0.05_0.05/test/**/*",
         outpath:str="../datasets/laion_inpainted_v2/laion_subset_10k_splits/test",
         srcpath:str="../datasets/laion_real/laion-aestheticsv2-6.5plus/**/*",
         gpu:int=0):
    
    # collect src paths for info
    allpaths = glob.glob(srcpath)
    print(len(allpaths))
    
    device = torch.device("cuda", gpu)
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float32
    ).to(device)

    img_extensions = [".png", ".jpeg", ".jpg", ".webp"]
    datapoints = {}
    for path in tqdm.tqdm(allpaths):
        basename = os.path.basename(path)
        imgid, extension = os.path.splitext(basename)
        if imgid not in datapoints:
            datapoints[imgid] = {}
        if extension.lower() in img_extensions:
            datapoints[imgid]["image"] = path
        elif extension == ".json":
            with open(path) as f:
                datapoints[imgid]["caption"] = json.load(f)["caption"]
                
    src_datapoints = datapoints
                
    # collect input paths for actual starting images
    allpaths = glob.glob(inpath)
    print(len(allpaths))
    img_extensions = [".png", ".jpeg", ".jpg", ".webp"]
    datapoints = {}
    for path in tqdm.tqdm(allpaths):
        basename = os.path.basename(path)
        imgid, extension = os.path.splitext(basename)
        if imgid not in datapoints:
            datapoints[imgid] = {}
        if extension.lower() in img_extensions:
            datapoints[imgid]["image"] = path
        elif extension == ".json":
            with open(path) as f:
                datapoints[imgid]["caption"] = json.load(f)["caption"]
                
    # merge captions from srcpaths into inpaths if inpaths doesn't have any
    for k, v in datapoints.items():
        if "caption" not in v:
            v["caption"] = src_datapoints[k]["caption"]
                
    allkeys = list(datapoints.keys())
    random.shuffle(allkeys)
    print(datapoints[allkeys[0]])

    outputdir = outpath
    os.makedirs(outputdir, exist_ok=True)
    for k in allkeys:
        datapoint = datapoints[k]
        out = generate_random_mask(PIL.Image.open(datapoint["image"]))
        masked_image, mask, comp_mask, original_image = out
        caption = datapoint["caption"]
        
        # inpaint
        inpainted = pipe(prompt=caption, image=masked_image, mask_image=mask).images[0]
        inpainted = PIL.Image.composite(inpainted, masked_image, comp_mask)
        
        #fake inpaint
        fakeinpainted = fake_inpaint(original_image, comp_mask, device=device, vae=pipe.vae)
        
        inpainted.save(os.path.join(outputdir, k+".realfake.webp"), lossless=True)
        fakeinpainted.save(os.path.join(outputdir, k+".fakefake.webp"), lossless=True)
        comp_mask.save(os.path.join(outputdir, k+".mask.webp"), lossless=True)
        

if __name__ == "__main__":
    fire.Fire(main)