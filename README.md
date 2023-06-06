# Tasks:
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

Evaluation metric: IoU
Loss function: Focal loss



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
