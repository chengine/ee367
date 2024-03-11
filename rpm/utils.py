import numpy as np
from ..ellipsoid_math.math_utils import *
import partial
import torch
from scipy.spatial import KDTree

class RPM():
    def __init__(self, env, epsilon=1e-3, kd_tree_radius=1e-2) -> None:
        self.rot_A = env['rotations']    # (batch x ndim x ndim)
        self.scale_A = env['scales']     # (batch x ndim)
        self.mu_A = env['means']         # (batch x ndim)

        self.ndim = self.mu_A.shape[-1]
        self.epsilon = epsilon           # margin around splats

        self.kdtree = KDTree(self.mu_A)
        self.kdtree_radius = kd_tree_radius
        bounding_poly, _ = sphere_to_poly(10, self.kdtree_radius, np.zeros(3))
        self.bounding_poly_A, self.bounding_poly_b = bounding_poly.A, bounding_poly.b

        self.ss = torch.linspace(0., 1., 100).cuda()[1:-1].reshape(1, -1)        # 1 x 100

        self.rot_B = torch.eye(self.ndim)
        self.scale_B = self.epsilon**2 * torch.ones(self.ndim)

    def compute_free_space(self, seed_pt=None, num_iters=100):
        # Would be nice if seed_pt is unoccupied

        # If seed_pt is provided, query a polytope 

        # Calculate the vertices of the polytope

        # Store the seed_pt, polytope, and vertices as a node and add to the graph

        # From each of the vertices of the polytope, query new polytopes
        # Make sure to not include vertices that are part of any upstream polytopes. TODO: Sufficient but not necessary condition
        

    def expand_nodes(self):
        

    def compute_poly(self, pts):
        # Computes the polytopes for a certain number of points. Returns list of A matrix and b vector
        inds = self.kdtree.query_ball_point(pts, self.kdtree_radius, return_sorted=True, workers=-1)

        b_offset = self.bounding_poly_A @ pts.T
        bounding_polys_b = (self.bounding_poly_b[..., None] + b_offset).T        # Num_pts in path x num_constraints

        # Create index list
        len_list = [len(ind) for ind in inds]
        valid_inds = [ind for ind in inds if len(ind) > 0]

        collision_pts = pts[np.array(len_list) > 0]

        if len(valid_inds) > 0:
            # print('Performing collision checks...')
            # At least one point needs checking
            len_valid_inds = [len(ind) for ind in valid_inds]
            valid_inds = np.concatenate(valid_inds)

            # Need to repeat the test points appropriately
            test_pts = [[collision_pts[i]]*length_i for i, length_i in enumerate(len_valid_inds)]
            test_pts = torch.from_numpy(np.concatenate(test_pts, axis=0)).to(device=device, dtype=torch.float32)

            rots = self.rot_A[valid_inds]
            scales = self.scale_A[valid_inds]
            means = self.mu_A[valid_inds]
        
            A_basic, b_basic = compute_supporting_hyperplanes_fast(rots, scales, self.rot_B, self.scale_B, means, pts, self.ss, 1.)

            As = []
            bs = []
            count = 0
            for i, pt in enumerate(pts):
                if len_list[i] > 0:
                    As_ = A_basic[count:count + len_list[i]]
                    bs_ = b_basic[count:count + len_list[i]]
        
                    A = np.concatenate([As_.reshape(-1, test_pts.shape[-1]), self.bounding_poly_A], axis=0)
                    b = np.concatenate([bs_.reshape(-1, ), bounding_polys_b[i]], axis=0)

                    A, b = h_rep_minimal(A, b, pt)

                    As.append(A)
                    bs.append(b)

                    count += len_list[i]
                else:
                    As.append(self.bounding_poly_A)
                    bs.append(bounding_polys_b[i])

        return As, bs