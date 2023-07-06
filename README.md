# Task Re-Definition and documentation (15.06)

## Task 1: Examine Baseline Performance with Different Inputs
- Investigate how blurred masks affect the performance of the baseline model.
  - The processing of the during training affects the later performance of the model. Mask handling ways:
    - 1. No specific handling of the mask, the mask image was loaded, one channel was extracted and treaded as a binary mask: test IoU 0.827
    - 2. Everything that is not 0 in the mask image will become 1 in the mask: test IoU 0.824
    - 3. Everything that is not 255 in the mask image will become 0 in the mask: **test IoU 0.944**
- Investigate how the baseline responds when 'fakefake' images are used as input.
  - The model still predicts very accuratly the mask area. To interpret this, we need to understand what is going on when inputting the mask and original image into the autoencoder to produce the fakefake images.
  - It seems like the Step to inpaint the mask via stable diffusion is not necessary in order to produce more data.
  - Next we can try to train on fakefake and the mask and see if we get comparable results to the baseline.

## Task 2: Dataset Generation
- Generate 10,000 images, each including 'original', 'fakefake', and 'realfake' images along with their corresponding masks.

## Task 3: Baseline training experiments to understand data effects
- Train on realfake (10k laion):
  - How does it perform when evaluated on fakefake images?
    - Test IoU: 0.934 (mask handling 3, on laion test set)
  - How does it perform when evaluated on realfake images?
    - test IoU 0.944 (mask handling 3, on laion test set)
  - How does it perform when evaluated on output images?
- Train on fakefakes: 
  - How does it perform when evaluated on fakefake images?
    - Test IoU: 0.950 (mask handling 3, on laion test set)
  - How does it perform when evaluated on realfake images?
    - Test IoU: 0.783 (mask handling 3, on laion test set)
  - How does it perform when evaluated on output images?
- Train on inpainted output (without copying the inpainting to original image):
  - How does it perform when evaluated on fakefake images?
  - How does it perform when evaluated on realfake images?
  - How does it perform when evaluated on output images?

## Task 4: Experimentation with Pre-Trained Visual Transformer (ViT) Based Models
- Train a pre-trained Visual Transformer (ViT) based model, possibly the Masked Autoencoder version (MAE-ViT) available on Huggingface, to understand how its attention mechanism improves or hampers inpainting detection.
- Compare the performance of the ViT based model with the baseline and Nix-Model.

## Task 5: Validate generation of (high-frequency) Noise Residue
- Identify if we can visually perceive the noise residue as illustrated in the reference paper. And if so, is this the right input to the model? Denis thinks yes, the equation in the paper seems off.

## Task 6: Nix-Model Validation and Training
- Validate the current implementation of the Nix-Model.
- Train the Nix-Model using the available 10,000 images. Compare its test IOU performance with the baseline.
- Train on realfake (10k laion):
  - How does it perform when evaluated on fakefake images?
    - Test IoU: 0.916 (NIX 40 million weight)
  - How does it perform when evaluated on realfake images?
    - test IoU 0.883 (NIX 40 million weight)
- Train on fakefakes: 
  - How does it perform when evaluated on fakefake images?
    - Test IoU: 0.905 (NIX 40 million weight)
  - How does it perform when evaluated on realfake images?
    - Test IoU: 0.759 (NIX 40 million weight)

## Task 7: Training and Testing Models on Combined Dataset
- Train the baseline model on the combined dataset and evaluate its performance.
- Similarly, train the Nix-Model on the combined dataset and evaluate its performance.
- Train the ViT based model on the combined dataset and evaluate its performance.
- Compare the performance of the three models.

## Task 8: Model Extrapolation
- Investigate how well the trained model extrapolates to other inpainting models. For instance, if a model is trained on SD1.5 data, can it detect when SD2.1 or Kandinsky was used for inpainting?



# Status June 13th
What we have done so far:
- Baseline is trained and evaluated
  - Pretrained FCN model from torchvision is used
  - test IoU is 0.827
  - Qualitative results are good
- We have code to produce more realistic masks
- Nix-Model is imeplemendet but training is not working yet (it does not seem to learn / output is always the same)
  - SRM seems to be wrong (potential issue)

## Current Methodology for Dataset Generation:

- **Original Image**: This refers to a specially generated image that is only available to Denis.

**Generation of "fakefake" Images**:
1. A combination of the original image and a mask is input into an Autoencoder.
2. The resulting image from the Autoencoder is the so-called **fakefake** image.

**Generation of "realfake" Images**:
1. The original image, along with the mask and the associated prompt, is subjected to a Stable Diffusion method.
2. The resulting image is an image **X**, which looks similar to the original but has altered areas within the mask region.
3. This image **X** is cropped along the mask and superimposed onto the original image to produce the **realfake** image.

## Planned Process for Dataset Generation:

- **Original Image**: For instance, an image from the COCO dataset, along with an associated prompt.

**Generation of a Mask**:
A mask is generated using one of three methods:
1. YOLO-based
2. Random-based
3. ResNet Panoptic-based

**Generation of "fakefake" Images**:
1. A combination of the original image and a mask is input into an Autoencoder.
2. The resulting image from the Autoencoder is the so-called **fakefake** image.

**Generation of "realfake" Images**:
1. The original image, along with the mask and the associated prompt, is subjected to a Stable Diffusion method.
2. The resulting image is an image **X**, which looks similar to the original but has altered areas within the mask region.
3. This image **X** is cropped along the mask and superimposed onto the original image to produce the **realfake** image.

## Tasks 13.06
- Was sagt baseline wenn wir fakefake bilder als input nehmen?
- Investigate how blurred masks affect the baseline
- Fix Noise resediue: Können wir den Noise resediue sehen wie im paper?
- Fix Nix-Model: Validiere implementierung
- Train Nix-Model auf den 10k bildern die wir schon haben, wie verhält sich test iou zu der von baseline?
- Generiere 10k bilder: jeweils: original, fakefake und realfake bilder (+ maske))
- Train baseline on combined dataset and test it
- Train Nix-Model on combined dataset and test it
- What is better? Baseline or Nix-Model?

**e-mail 13.06**
I was checking the NIX paper again and I'm still not sure why they chose to write down R_i like in Eq. 3 but I think what R_i should be is the (high-frequency) noise content of the image, like shown in the images as well.
I also wanted to add that it would be cool to try a pre-trained visual transformer (ViT) based model (maybe the masked autoencoder version, MAE-ViT, it's on huggingface) since its attention could enable better comparisons across the entire image (but it could also be disadvantageous in other respects, like the larger number of parameters).

Regarding the questions that we would like to answer, you should eventually run experiments including the following (it's mostly a summary of what we discussed, plus a few additional points):
1. NIX vs simple FCN baseline (vs fine-tuned ViT baseline)
2. using fake fake (which would actually be similar to NIX's "universal" data generation) vs real fakes.
3. I'm also really curious how training using the inpainted output directly works (so without pasting the inpainted region into the original image). If you train like this, how does it work on fake-fakes and real-fakes?
4. how well does the trained model extrapolate to other inpainting models? For example, can your model trained on SD1.5 data detect when SD2.1 or Kandinsky was used for inpainting?


# Tasks (6.6):
- **Train baseline**: 
  - Use pretrained FCN model from torchvision
  - input: realfake images
    - without additional noise input?
  - target: masked image
  - Goal: Predict the inpainted area just by looking at the image

- **Prdouce additional data**:
  - implement relaistic mask generation process
    - Possibly by using YOLO to detect objects and then segmenting them out
    - Possibly by generating random polygons
  - generate image with stable diffusion --> fakefake images
  - use variational autoencoder to produce encoded-decoded image
  - fuse fakefake with encoded-decoded image where mask is at

- **implement NIX-NET from paper**
  - implement NIX-NET from paper "Noise doesn't lie" paper
  - train the model:
    - input: realfake images
    - input: image noise
    - target: masked image

- Evaluation metric: IoU
- Loss function: Focal loss



# Topic 2: Detecting Inpainting

## Goal
Given an image, detect which parts of it have been inpainted using some inpainting model, with a focus on diffusion-based methods.

## Motivation
Generative models can be used not only for generating novel images from scratch but also to modify existing real images, such as by adding or removing objects. These modifications can be used as a means of disinformation, making it crucial to detect whether and which regions of an image have been modified using diffusion-based inpainting methods.

## [NEW] First Goal (13 June)
Train a simple CNN-based network with inpainting data.

### Model
You can use a pre-trained Fully Convolutional Network (FCN) or other models available in the [PyTorch Vision models](https://pytorch.org/vision/stable/models.html#semantic-segmentation) (or elsewhere).

### Data
The dataset can be found at: [https://ruhr-uni-bochum.sciebo.de/f/1777053235](https://ruhr-uni-bochum.sciebo.de/f/1777053235) (only square masks). Additional data can be obtained online.

- Files ending in `*realfake.webp` are inpainted using the entire model.
- Files ending in `*fakefake.webp` are re-encoded using the latent autoencoder and then blended using the mask (`*.mask.webp`).

### Loss
Use focal loss as the loss function.

### Implementation and Evaluation
- Use Intersection over Union (IoU) as the evaluation metric.
- Write code for inpainting with more realistic masks.
- Write code for inpainting with Stable Diffusion.
- Adapt and use the provided code as a starting point: [https://gist.github.com/lukovnikov/35ca3fac8449ad6b3f51d356699f27dc](https://gist.github.com/lukovnikov/35ca3fac8449ad6b3f51d356699f27dc).
- Implement more realistic masks following this example: [https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_ops.py#L156](https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_ops.py#L156).

## Description
- Conduct a literature study on inpainting techniques.
- Obtain a dataset of inpainted images (a few thousand images are currently available, more can be generated).
- Implement a baseline method, such as "Noise doesn't lie" (refer to [https://www.ijcai.org/proceedings/2021/0109.pdf](https://www.ijcai.org/proceedings/2021/0109.pdf)), and evaluate its performance.
- Identify limitations and attempt to break the baseline method.
- Continuously improve the detection approach.
- Develop a unified approach for fake detection and inpainting detection.
