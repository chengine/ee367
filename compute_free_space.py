#%%
import torch
from ellipsoid_math.math_utils import *
import time
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.random.manual_seed(0)
np.random.seed(0)

def rot_z(thetas):
    row1 = torch.stack([torch.cos(thetas), -torch.sin(thetas)], dim=-1)
    row2 = torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=-1)

    return torch.stack([row1, row2], dim=1)

n = 20

thetas = torch.rand(n).cuda()
mu_A = torch.rand(n, 2).cuda()
rot_A = rot_z(thetas)
scale_A = 0.2*torch.rand(n, 2).cuda()
sqrt_sigma_A = torch.bmm(rot_A, torch.diag_embed(scale_A))
Sigma_A = torch.bmm(sqrt_sigma_A, sqrt_sigma_A.transpose(2, 1))
scale_A = scale_A**2

rot_B = rot_z(torch.zeros(1).cuda())
scale_B = 0.001 * torch.ones(1,2).cuda()

sqrt_sigma_B = torch.bmm(rot_B, torch.diag_embed(scale_B))
Sigma_B = torch.bmm(sqrt_sigma_B, sqrt_sigma_B.transpose(2, 1)).squeeze()
rot_B = rot_B.squeeze()
scale_B = scale_B.squeeze()**2

ss = torch.linspace(0., 1., 100).cuda()[1:-1].reshape(1, -1)        # 1 x 100

# test_type = 'gen_eig'
    
# if test_type == 'matmul':
#     traced_test = torch.jit.trace(compute_polytope_fast, (rot_A, scale_A, rot_B, scale_B, mu_A, mu_B, ss))
# elif test_type == 'gen_eig':
#     traced_test = torch.jit.trace(compute_polytope, (Sigma_A, Sigma_B, mu_A, mu_B))

### Tests the full space
X, Y = torch.meshgrid(torch.linspace(-0.25, 1.25, 10, device=device), torch.linspace(-0.25, 1.25, 10, device=device))
mu_B = torch.stack([X, Y], dim=-1).reshape(-1, 2)

bounding_A = np.concatenate([np.eye(2), -np.eye(2)], axis=0)
bounding_b = np.concatenate([1.25*np.ones(2), -(-0.25*np.ones(2))], axis=0)

polys = []
tnow = time.time()
for mu_b in mu_B:

    poly = compute_polytope_fast(rot_A, scale_A, rot_B, scale_B, mu_A, mu_b, ss, A_bound=bounding_A, b_bound=bounding_b)
    polys.append(poly)

print('Time to solve polytope computation: ', time.time() - tnow)

#%%
    
fig, ax = plt.subplots(figsize=(10,10), dpi=100)

for mu_A_, Sigma_A_ in zip(mu_A, Sigma_A):
    plot_ellipse(mu_A_, Sigma_A_, n_std_tau=1., ax=ax, facecolor='r', edgecolor='r', alpha=0.5, linewidth=1)

# plot_halfplanes(poly.A, poly.b, -0.5, 1.5, ax)
for poly in polys:  
    try:
        plot_polytope(polytope.Polytope(poly[0], poly[1]), ax)
    except:
        print('Not a valid polytope')
        pass

test_pts = mu_B.cpu().numpy()
ax.scatter(test_pts[:, 0], test_pts[:, 1], color='black', alpha=0.3)

ax.set_xlim([-0.25, 1.25])
ax.set_ylim([-0.25, 1.25])
ax.axis('off')
# fig.savefig('teaser_filling.png', transparent=True)