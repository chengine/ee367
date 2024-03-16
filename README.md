# CineNeRF: Hallucinating the Best Shot
## This repo implements generative cinematography using Gaussian Splatting with knowledge distillation from the vision-language model CLIP to extract semantic information of the scene. Users must provide an image stream of a scene and train a Gaussian Splat
## using [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) At test time, the pipeline can accept text queries of where the camera should point at and where it should go to. The method will then associate these queries with relevant Gaussians
## and use the centroid of these relevant Gaussians to bias the camera trajectories. 

## Installation Instructions

## The installation instructions assumes you have installed [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) from source. You must additionally install [gsplat](https://github.com/nerfstudio-project/gsplat) to be able to use Gaussian Splatting with Nerfstudio.

### 1. Clone this repo.
`git clone git@github.com:shorinwa/gemsplat.git`

### 2. Install `gemsplat` as a python package.
`python -m pip install -e .`

## 3. Register `gemsplat` with Nerfstudio.
`ns-install -cli`

### Now, you can run `gemsplat` like other models in Nerfstudio using the `ns-train gemsplat` command.
