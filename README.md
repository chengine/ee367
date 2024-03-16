# CineNeRF: Hallucinating the Best Shot
## This repo implements generative cinematography using Gaussian Splatting with knowledge distillation from the vision-language model CLIP to extract semantic information of the scene. Users must provide an image stream of a scene and train a Gaussian Splat using [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) At test time, the pipeline can accept text queries of where the camera should point at and where it should go to. The method will then associate these queries with relevant Gaussians and use the centroid of these relevant Gaussians to bias the camera trajectories. Videos can be found at our [Google Drive](https://drive.google.com/drive/u/0/folders/1H4DUR7NeCqm-TfHIMmKufGwOiaLYA4tL), where additionally our training data and model parameters are in the Data folder. Our report can be found [here](CineNeRF.pdf).

## Installation Instructions

## The installation instructions assumes you have installed [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) from source. You must additionally install [gsplat](https://github.com/nerfstudio-project/gsplat) to be able to use Gaussian Splatting with Nerfstudio. The commit of Nerfstudio used in this work is on Feb. 1.

### 1. Go into the `sagesplat` folder and install `sagesplat` as a python package. Before you do that, you must install SegmentAnything. Disclaimer: We don't actually use SegmentAnything in this work, but `sagesplat` is part of a broader research effort. 
`pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git`
`python -m pip install -e .`

## 2. Register `sagesplat` with Nerfstudio.
`ns-install -cli`

### Now, you can run `sagesplat` like other models in Nerfstudio using the `ns-train sagesplat` command. 

## The rest of the dependencies include: [dijkstra3d](https://github.com/seung-lab/dijkstra3d), [polytope], [cvxopt], and [cvxpy].

## Run `eval.py` for the main script. Things to change for use is: (1) path to Nerfstudio config file (Line 147), the bounds of the scene (optional, Line 171), the semanantic text queries (Line 182, 186), the squared radius of the camera volume (Line 210). You can also change the initial point (Line 220) if you don't want it to be random.
