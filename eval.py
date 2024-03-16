# %%
from __future__ import annotations

import json
import os
from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Union, Tuple

from typing_extensions import Annotated
import pickle
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib as mpl
from tqdm import tqdm
import open3d as o3d
from scipy.spatial import KDTree

from utils.nerf_utils import *
from ellipsoid_math.math_utils import *
from gs_utils.gs_utils import *
from spline_utils.spline_utils import *
from grid_utils.init_path import *

#%% Functions
def compute_centroid(points, values):

    return np.sum(values[..., None]*points, axis=0) / np.sum(values)

def mask_attributes(attr, mask):

    new_attr = {}

    for key, value in attr.items():
        new_attr[key] = value[mask]

    return new_attr

def query_semantics(query, nerf, threshold=0.85):
    # list of positives
    # ARMLAB: ['hammer', 'coffeecup', 'mug', 'drill', 'cup']
    positives = query

    # update list of negatives ['things', 'stuff', 'object', 'texture']: 'object, things, stuff, texture'
    negatives = 'object, things, stuff, texture'

    # get the semantic outputs
    semantic_info = nerf.get_semantic_point_cloud(positives=positives,
                                                    negatives=negatives,
                                                    pcd_attr=env_attr)

    # scaled similarity
    sc_sim = torch.clip(semantic_info["similarity"] - 0.5, 0, 1)
    sc_sim = sc_sim / (sc_sim.max() + 1e-6)

    # threshold for masking the point cloud
    sim_mask = (sc_sim > threshold).cpu().numpy().squeeze()

    return sc_sim, sim_mask, semantic_info

def get_point_cloud(points, values, mask=None):

    if mask is not None:
        points = points[mask]
        values = values[mask]

    if values.shape[-1] == 3:
        colors = values
    elif values.shape[-1] > 3:
        colors = values[:, :3]
    elif values.shape[-1] == 1:
        colors = mplcm.turbo(values)[:, :3]
    else:
        raise ValueError('Incorrect dimensions for color value.')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(values)

    return pcd

def create_gs_mesh(attr, cs=None, res=4, probabilistic_factor=None):
    
    means = attr['means'].cpu().detach().numpy()
    rotations = quaternion_to_rotation_matrix(attr['quats']).cpu().detach().numpy()
    scales = attr['scales'].cpu().detach().numpy()
    opacities = attr['opacities'].cpu().detach().numpy()
    colors = attr['colors'].cpu().detach().numpy()

    if cs is None:
        cs = colors
    else:
        if cs.shape[-1] == 3:
            cs = cs
        elif cs.shape[-1] > 3:
            cs = cs[:, :3]
        elif cs.shape[-1] == 1:
            cs = mplcm.turbo(cs)[:, :3]
        else:
            raise ValueError('Incorrect dimensions for color value.')
        
    if isinstance(probabilistic_factor, int):
        scales = (opacities.squeeze() * probabilistic_factor)[:, None] * scales

    scene = o3d.geometry.TriangleMesh()

    for i, (mean, R, S, col) in enumerate(zip(means, rotations, scales, cs)):
        one_gs_mesh = o3d.geometry.TriangleMesh.create_sphere(resolution=res)
        points = np.asarray(one_gs_mesh.vertices)
        new_points = points * S[None]
        one_gs_mesh.vertices = o3d.utility.Vector3dVector(new_points)
        one_gs_mesh = one_gs_mesh.paint_uniform_color(col)
        one_gs_mesh = one_gs_mesh.rotate(R)
        one_gs_mesh = one_gs_mesh.translate(mean)
        scene += one_gs_mesh

    return scene

def point_camera_pose(position, look_at):
    # Convention is x right, y up, z back 

    # We want the z-axis to look away at the look_at point

    z = (position - look_at) 
    z = z / np.linalg.norm(z)

    # we want the y axis to be as close to the real z-axis as possible (meaning drone is close to hover)
    true_z = np.array([0, 0, 1.])
    x = np.cross(true_z, z)

    y = np.cross(z, x)

    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, y, z], axis=-1)
    pose[:3, 3] = position

    return pose
# %%
# # # # #
# # # # # Config Path
# # # # #

config_path = Path('outputs/armlab/sagesplat/2024-03-10_011847/config.yml')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize NeRF
nerf = NeRF(config_path=config_path,
            res_factor=None,
            test_mode="test", # Options: "inference", "val", "test"
            dataset_mode="train",
            device=device)

# camera intrinsics
H, W, K = nerf.get_camera_intrinsics()
K = K.to(device)

# poses in test dataset
eval_poses = nerf.get_poses()

# images for evaluation
eval_imgs = nerf.get_images()

#%% generate the point cloud of the environment

bound = np.array([[-0.75, 0.75], [-0.75, 0.75], [-0.5, 0.5]])

gsgrid = GS_Grid(nerf, bound, N_dis=100)
occupied_grid = gsgrid.compute_gs_grid()

env_pcd, env_pcd_mask, env_attr = nerf.generate_point_cloud(use_bounding_box=True, bounding_box_min=bound[:, 0], bounding_box_max=bound[:, 1])

# o3d.visualization.draw_plotly([env_pcd])
# %%
# Semantic Query

look_at = 'kettle'
look_at_similarity, look_at_mask, _ = query_semantics(look_at, nerf, threshold=0.85)
look_at_attr = mask_attributes(env_attr, look_at_mask)

go_to = 'apple'
go_to_similarity, go_to_mask, _ = query_semantics(go_to, nerf, threshold=0.85)
go_to_attr = mask_attributes(env_attr, go_to_mask)

# create the point cloud
look_at_mesh = create_gs_mesh(look_at_attr, cs=None, res=4, probabilistic_factor=2)
go_to_mesh = create_gs_mesh(go_to_attr, cs=None, res=4, probabilistic_factor=2)
# o3d.visualization.draw_geometries([look_at_mesh, go_to_mesh])

# env_mesh = create_gs_mesh(env_attr, cs=None, res=3, probabilistic_factor=2)

o3d.io.write_triangle_mesh(f'look_at_{look_at}.obj', look_at_mesh, print_progress=True)
o3d.io.write_triangle_mesh(f'go_to_{go_to}.obj', go_to_mesh, print_progress=True)
# o3d.io.write_triangle_mesh(f'env.obj', env_mesh, print_progress=True)

#%%

look_at_center = compute_centroid(look_at_attr['means'].detach().cpu().numpy(), (look_at_similarity[look_at_mask].detach().cpu().numpy() * look_at_attr['opacities'].detach().cpu().numpy()).squeeze())

go_to_center = compute_centroid(go_to_attr['means'].detach().cpu().numpy(), (go_to_similarity[go_to_mask].detach().cpu().numpy() * go_to_attr['opacities'].detach().cpu().numpy()).squeeze())

# Construct KD-tree of entire scene
kdtree = KDTree(gsgrid.means.cpu().numpy())
radius = .1
kappa = 0.03**2     # 0.03 radius of robot
tau = 1.

# Instantiate sphere corresponding to KDTree ball radius
bounding_poly, sphere_pts = sphere_to_poly(10, radius, np.zeros(3))
bounding_poly_A, bounding_poly_b = bounding_poly.A, bounding_poly.b

spline_planner = SplinePlanner()

# Random starting point
s0 = np.random.rand(3)*(bound[:, 1] - bound[:, 0]) + bound[:, 0]
# s0[-1] = -.2
# s0 = np.array([-0.7, 0.5, -0.2])
# s0 = np.array([0., -0.5, -0.2])

# Test if this s0 point is collision-free
inds = kdtree.query_ball_point(s0, radius, return_sorted=True, workers=-1)

if len(inds) > 0:
    b_offset = bounding_poly_A @ s0
    bounding_polys_b = (bounding_poly_b + b_offset)

    R = gsgrid.rotations[inds]
    D = gsgrid.prob_scalings[inds]**2       # NOTE: Need to square the scaling matrix to get eigenvalues
    mu_A = gsgrid.means[inds]
    A0, b0 = compute_polytope(R, D, kappa, mu_A, torch.from_numpy(s0).to(device=device, dtype=torch.float32), tau, bounding_poly_A, bounding_polys_b)

    # try:
    #     s0 = check_and_project(A0, b0.squeeze(), s0)
    #     assert s0 is not None
    # except:
    #     raise AssertionError('Initial point is in collision. Please try again.')

goal = go_to_center

# Solve for path
astar = Astar(gsgrid.gs_grid.cpu().numpy(), gsgrid.grid_pts.cpu().numpy())
path = astar.create_path(s0, goal)

polys = []

inds = kdtree.query_ball_point(path, radius, return_sorted=True, workers=-1)
b_offset = bounding_poly_A @ path.T
bounding_polys_b = (bounding_poly_b[..., None] + b_offset).T        # Num_pts in path x num_constraints

# Create index list
len_list = [len(ind) for ind in inds]
valid_inds = [ind for ind in inds if len(ind) > 0]

collision_pts = path[np.array(len_list) > 0]

if len(valid_inds) > 0:
    # print('Performing collision checks...')
    # At least one point needs checking
    len_valid_inds = [len(ind) for ind in valid_inds]
    valid_inds = np.concatenate(valid_inds)

    # Need to repeat the test points appropriately
    test_pts = [[collision_pts[i]]*length_i for i, length_i in enumerate(len_valid_inds)]
    test_pts = torch.from_numpy(np.concatenate(test_pts, axis=0)).to(device=device, dtype=torch.float32)

    R = gsgrid.rotations[valid_inds]
    D = gsgrid.prob_scalings[valid_inds]**2       # NOTE: Need to square the scaling matrix to get eigenvalues
    mu_A = gsgrid.means[valid_inds]

    nominal_As, nominal_bs, _ = compute_supporting_hyperplanes(R, D, kappa, mu_A, test_pts, tau)

    any_pt_in_collision = gs_sphere_intersection_test(R, D, kappa, mu_A, test_pts, tau, return_raw=True)

    As = []
    bs = []
    unsafes = []
    count = 0
    for i, pt in enumerate(path):
        if len_list[i] > 0:
            As_ = nominal_As[count:count + len_list[i]]
            bs_ = nominal_bs[count:count + len_list[i]]
            pt_collision = any_pt_in_collision[count:count + len_list[i]]

            is_unsafe = ~torch.all(pt_collision)
            unsafes.append(is_unsafe)

            # if is_unsafe:
            #     n_occ = gsgrid.update_grid(torch.from_numpy(pt).to(device=device, dtype=torch.float32))

            A = np.concatenate([As_.reshape(-1, test_pts.shape[-1]), bounding_poly_A], axis=0)
            b = np.concatenate([bs_.reshape(-1, ), bounding_polys_b[i]], axis=0)

            if not is_unsafe:
                A, b = h_rep_minimal(A, b, pt)

            if not is_unsafe:
                As.append(A)
                bs.append(b)

            count += len_list[i]
        else:
            As.append(bounding_poly_A)
            bs.append(bounding_polys_b[i])

    unsafe = torch.any(torch.tensor(unsafes))

else:
    # print('No points in trajectory need checking...')
    # No points in trajectory need checking
    As = [bounding_poly_A for i in range(len(path))]
    bs = [bounding_polys_b[i] for i in range(len(path))]

    unsafe = False

try:
    # Then attempt to solve a smooth b-spline path
    traj = spline_planner.optimize_b_spline(As, bs, s0, goal)

except:
    # B-spline program couldn't solve.
    print('Could not solve B-spline program')

#%%
# Compute 6DOF Pose Trajectory
imgs = []
poses = []
outs = []
for i, waypt in enumerate(traj[::3]):
    pose = point_camera_pose(waypt[:3], look_at_center)
    outputs = nerf.render(torch.from_numpy(pose).to(dtype=torch.float32))
    imgs.append(outputs['rgb'].cpu().numpy())
    poses.append(pose)
    outs.append(outputs)

    print(f'Rendered pose {i}')

# Create Video
fourcc = cv2.VideoWriter.fourcc(*'mp4v')  
video = cv2.VideoWriter(f"look_at_{look_at} go_to_{go_to}.avi",fourcc,60, (imgs[0].shape[1],imgs[0].shape[0]))
for img in imgs:
    video.write(cv2.cvtColor((255*img).astype(np.uint8), cv2.COLOR_RGB2BGR))
video.release()

directory = f'Figures/look_at_{look_at} go_to_{go_to}'
if not os.path.exists(directory):
    os.makedirs(directory)
for i, img in enumerate(imgs[::len(imgs)//8]):
    cv2.imwrite(directory + f'/{i}.png', cv2.cvtColor((255*img).astype(np.uint8), cv2.COLOR_RGB2BGR))

# %% Write traj

save_path = f"Figures/look_at_{look_at} go_to_{go_to}/poses.json"
data = {
    'poses': [pose.tolist() for pose in poses]
}

with open(save_path, 'w') as f:
    json.dump(data, f, indent=4)

# %%
