import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn

from matplotlib.transforms import Affine2D
from matplotlib.patches import Ellipse

# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def plot_gaussians(means, covariances, values, scale=1.0):
    n, d = means.shape

    means = means.detach().cpu().numpy()
    values = values.detach().cpu().numpy()
    covariances = covariances.detach().cpu().numpy()
    covariance = np.zeros((covariances.shape[0], 3, 3))
    covariance[:,0,0] = covariances[:,0]
    covariance[:,1,0] = covariances[:,1]
    covariance[:,0,1] = covariances[:,1]
    covariance[:,1,1] = covariances[:,2]
    covariance[:,2,2] = 1.0

    fig = plt.figure()
    ax = fig.gca()

    vmin = np.min(values)
    vmax = np.max(values)

    for i in range(n):
        color = matplotlib.cm.get_cmap("viridis")
        v = (values[i,0] - vmin) / vmax
        ellipse = Ellipse(
            xy=(0.0, 0.0), width=10.0, height=10.0, fc=color(v), alpha=0.25)
        affine = Affine2D(covariance[i]).translate(*means[i,:2])
        ellipse.set_transform(affine + ax.transData)

        ax.add_patch(ellipse)

    plt.axis((-scale * 1.25, scale * 1.25, -scale * 1.25, scale * 1.25))
    plt.gca().set_aspect("equal", adjustable="box")
    # plt.axis("scaled")
    # plt.axis("off")
    return fig

def sample_gaussians(means, conics, values, samples):
    n, d = means.shape

    x = samples.reshape(-1, 1, d, 1) - means.reshape(1, n, d, 1)
    powers = -0.5 * (x.transpose(-1, -2) @ (conics @ x))
    densities = torch.exp(powers).reshape(-1, n, 1)

    values = values.reshape(1, n, -1)
    res = densities * values

    return res.sum(1)

def region_kernel(size, dx, d):
    half_size = (size - 1) / 2
    t = torch.linspace(-half_size, half_size, size).cuda() * dx
    td = [t for i in range(d)]

    grid = torch.meshgrid(td, indexing="xy")
    return torch.stack(grid, dim=-1).reshape(-1, d)

def sample_gaussians_region(means, conics, values, center, size, dx):
    n, d = means.shape
    samples = _sample_gaussians_region(d, center, size, dx)
    return sample_gaussians(means, conics, values, samples)

def sample_gaussians_img(means, conics, values, w, h, scale):
    n, d = means.shape

    tx = torch.linspace(-1, 1, w).cuda() * scale
    ty = torch.flip(torch.linspace(-1, 1, h).cuda().unsqueeze(-1), (0,1)).squeeze() * scale
    gx, gy = torch.meshgrid((tx, ty), indexing="xy")
    if d == 3:
        gz = torch.ones((w,h), device="cuda")
        samples = torch.stack((gx, gy, gz), dim=-1).reshape(w * h, d)
    if d == 2:
        samples = torch.stack((gx, gy), dim=-1).reshape(w * h, d)

    img = sample_gaussians(means, conics, values, samples)

    return img.reshape(w, h, -1)

def gaussian_derivative(means, conics, values, samples):
    n, d = means.shape

    x = samples.reshape(-1, 1, d, 1) - means.reshape(1, n, d, 1)
    inv_prod = conics @ x
    powers = -0.5 * (x.transpose(-1, -2) @ inv_prod)
    densities = torch.exp(powers).reshape(-1, n, 1, 1)
    derivatives = -inv_prod.reshape(-1, n, d, 1) * densities

    values = values.reshape(1, n, 1, -1)
    res = derivatives * values

    return res.sum(1)

def gaussian_derivative2(means, conics, values, samples):
    n, d = means.shape

    x = samples.reshape(-1, 1, d, 1) - means.reshape(1, n, d, 1)
    inv_prod = conics @ x
    powers = -0.5 * (x.transpose(-1, -2) @ inv_prod)
    densities = torch.exp(powers).reshape(-1, n, 1, 1, 1)
    ones = torch.ones(x.shape, device="cuda")
    derivatives = (inv_prod @ inv_prod.transpose(-1, -2) - conics).reshape(-1, n, d, d, 1) * densities

    values = values.reshape(1, n, 1, 1, -1)
    res = derivatives * values

    return res.sum(1)

# def rasterize_gaussians(means, covariances, opacities, values, w, h):
#     projection = torch.tensor([
#         [1.0, 0.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0, 0.0],
#         [0.0, 0.0, 1.0, 0.0],
#         [0.0, 0.0, 0.0, 1.0]
#     ], device="cuda")
#     raster_settings = GaussianRasterizationSettings(
#         image_height = w,
#         image_width = h,
#         tanfovx = 16.0,
#         tanfovy = 16.0,
#         bg = torch.zeros(3, device="cuda"),
#         scale_modifier = 1.0,
#         viewmatrix = torch.diag(torch.ones(4, device="cuda")),
#         projmatrix = projection,
#         sh_degree = 0,
#         campos = torch.zeros(3, device="cuda"),
#         prefiltered = False,
#         debug = False,
#     )
#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)
#     means3D = means.reshape(-1, 3)
#     means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
#     covariances = covariances.reshape(-1, 3, 3)
#     cov3D = torch.zeros((covariances.shape[0], 6), dtype=means3D.dtype, device="cuda")
#     cov3D[:,0] = covariances[:,0,0]
#     cov3D[:,1] = covariances[:,0,1]
#     cov3D[:,2] = covariances[:,0,2]
#     cov3D[:,3] = covariances[:,1,1]
#     cov3D[:,4] = covariances[:,1,2]
#     cov3D[:,5] = covariances[:,2,2]
# 
#     rendered_image, radii = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = None,
#         colors_precomp = values.reshape(-1, 3),
#         opacities = opacities.reshape(-1, 1),
#         scales = None,
#         rotations = None,
#         cov3D_precomp = cov3D)
# 
#     return rendered_image.transpose(0, -1)

def build_full_covariances(s, t):
    t = torch.tanh(t) * s.prod(-1).sqrt().unsqueeze(-1)
    S = torch.diag_embed(s)
    # T = torch.diag_embed(torch.zeros(s.shape, device="cuda"))
    # c = torch.cos(t).squeeze()
    # s = torch.sin(t).squeeze()
    # T[:,0,0] = c
    # T[:,1,1] = c
    # T[:,0,1] = -s
    # T[:,1,0] = s
    indices = torch.tril_indices(s.shape[-1], s.shape[-1], -1)

    S[..., indices[0], indices[1]] = t
    S[..., indices[1], indices[0]] = t
    # T = T + torch.diag_embed(torch.ones(s.shape, device="cuda"))

    # covariances = T @ S @ torch.transpose(T, -1, -2)
    covariances = S
    conics = torch.inverse(covariances)

    return covariances, conics
    # return T @ S

def flatten_covariances(covariances, conics):
    covariances = covariances.reshape(*covariances.shape[:-2], -1)[..., [0, 1, 3]]
    conics = conics.reshape(*conics.shape[:-2], -1)[..., [0, 1, 3]]
    return covariances, conics

def build_covariances(s, t):
    covariances, conics = build_full_covariances(s, t)
    return flatten_covariances(covariances, conics)

import unittest

class UnitTests(unittest.TestCase):
    def contains(self, expected, samples):
        for element in expected:
            anyTrue = False
            for sample in samples:
                anyTrue |= torch.all(torch.isclose(sample, element.cuda()))

            self.assertTrue(anyTrue)

    def test_sample_region_2d_2(self):
        samples = region_kernel(2, 1.0, 2)
        self.assertEqual(samples.shape, (4, 2))
        expected = [
            torch.tensor([-0.5, -0.5]),
            torch.tensor([0.5, -0.5]),
            torch.tensor([-0.5, 0.5]),
            torch.tensor([0.5, 0.5]),
        ]
        self.contains(expected, samples)

    def test_sample_region_2d_3(self):
        samples = region_kernel(3, 0.5, 2)
        self.assertEqual(samples.shape, (9, 2))
        expected = [
            torch.tensor([-0.5, 0.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.5, 0.0]),
            torch.tensor([-0.5, 0.5]),
            torch.tensor([0.0, 0.5]),
            torch.tensor([0.5, 0.5]),
            torch.tensor([-0.5, -0.5]),
            torch.tensor([0.0, -0.5]),
            torch.tensor([0.5, -0.5]),
        ]
        self.contains(expected, samples)

    def test_sample_region_3d_2(self):
        samples = region_kernel(2, 2.0, 3)
        self.assertEqual(samples.shape, (8, 3))
        expected = [
            torch.tensor([1.0, -1.0, -1.0]),
            torch.tensor([1.0, -1.0, -1.0]),
            torch.tensor([1.0, 1.0, -1.0]),
            torch.tensor([1.0, 1.0, -1.0]),
            torch.tensor([1.0, -1.0, 1.0]),
            torch.tensor([1.0, -1.0, 1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        ]
        self.contains(expected, samples)

if __name__ == '__main__':
    unittest.main()
