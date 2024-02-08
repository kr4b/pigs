import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as f

from torch import nn

from matplotlib.transforms import Affine2D
from matplotlib.patches import Ellipse

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def plot_gaussians(means, covariances, opacities, values):
    n, d = means.shape

    means = means.cpu().numpy()
    values = values.cpu().numpy()
    opacities = opacities.cpu().numpy()
    covariances = covariances.detach().cpu().numpy()
    covariance = np.zeros((covariances.shape[0], 3, 3))
    covariance[:,:d,:d] = covariances
    covariance[:,2,2] = 1.0

    fig = plt.figure()
    ax = fig.gca()

    for i in range(n):
        color = matplotlib.cm.get_cmap("viridis")
        ellipse = Ellipse(xy=(0.0, 0.0), width=25.0, height=25.0, fc=color(values[i,0]), alpha=opacities[i].item())
        affine = Affine2D(covariance[i]).translate(*means[i,:2])
        ellipse.set_transform(affine + ax.transData)

        ax.add_patch(ellipse)

    plt.axis("scaled")
    # plt.axis("off")
    return fig

def sample_gaussians(means, conics, opacities, values, samples):
    n, d = means.shape

    x = samples.reshape(-1, 1, d, 1) - means.reshape(1, n, d, 1)
    powers = -0.5 * (x.transpose(-1, -2) @ (conics @ x))
    densities = torch.exp(powers).reshape(-1, n, 1)

    opacities = opacities.reshape(1, n, 1)
    values = values.reshape(1, n, -1)

    res = densities * opacities * values

    return res.sum(1)

def region_kernel(size, dx, d):
    half_size = (size - 1) / 2
    t = torch.linspace(-half_size, half_size, size).cuda() * dx
    td = [t for i in range(d)]

    grid = torch.meshgrid(td, indexing="xy")
    return torch.stack(grid, dim=-1).reshape(-1, d)

def sample_gaussians_region(means, conics, opacities, values, center, size, dx):
    n, d = means.shape
    samples = _sample_gaussians_region(d, center, size, dx)
    return sample_gaussians(means, conics, opacities, values, samples)

def sample_gaussians_img(means, conics, opacities, values, w, h, scale):
    n, d = means.shape

    tx = torch.linspace(-1, 1, w).cuda() * scale
    ty = torch.flip(torch.linspace(-1, 1, h).cuda().unsqueeze(-1), (0,1)).squeeze() * scale
    gx, gy = torch.meshgrid((tx, ty), indexing="xy")
    if d == 3:
        gz = torch.ones((w,h), device="cuda")
        samples = torch.stack((gx, gy, gz), dim=-1).reshape(w * h, d)
    if d == 2:
        samples = torch.stack((gx, gy), dim=-1).reshape(w * h, d)

    img = sample_gaussians(means, conics, opacities, values, samples)

    return img.reshape(w, h, -1)

def gaussian_derivative(means, conics, opacities, values, samples):
    n, d = means.shape

    x = samples.reshape(-1, 1, d, 1) - means.reshape(1, n, d, 1)
    inv_prod = conics @ x
    powers = -0.5 * (x.transpose(-1, -2) @ inv_prod)
    densities = torch.exp(powers).reshape(-1, n, 1, 1)
    derivatives = -inv_prod.reshape(-1, n, d, 1) * densities

    opacities = opacities.reshape(1, n, 1, 1)
    values = values.reshape(1, n, 1, -1)

    res = derivatives * opacities * values

    return res.sum(1)

def gaussian_derivative2(means, conics, opacities, values, samples):
    n, d = means.shape

    x = samples.reshape(-1, 1, d, 1) - means.reshape(1, n, d, 1)
    inv_prod = conics @ x
    powers = -0.5 * (x.transpose(-1, -2) @ inv_prod)
    densities = torch.exp(powers).reshape(-1, n, 1, 1, 1)
    ones = torch.ones(x.shape, device="cuda")
    derivatives = (inv_prod @ inv_prod.transpose(-1, -2) - conics).reshape(-1, n, d, d, 1) * densities

    opacities = opacities.reshape(1, n, 1, 1, 1)
    values = values.reshape(1, n, 1, 1, -1)

    res = derivatives * opacities * values

    return res.sum(1)

def rasterize_gaussians(means, covariances, opacities, values, w, h):
    projection = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], device="cuda")
    raster_settings = GaussianRasterizationSettings(
        image_height = w,
        image_width = h,
        tanfovx = 16.0,
        tanfovy = 16.0,
        bg = torch.zeros(3, device="cuda"),
        scale_modifier = 1.0,
        viewmatrix = torch.diag(torch.ones(4, device="cuda")),
        projmatrix = projection,
        sh_degree = 0,
        campos = torch.zeros(3, device="cuda"),
        prefiltered = False,
        debug = False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = means.reshape(-1, 3)
    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
    covariances = covariances.reshape(-1, 3, 3)
    cov3D = torch.zeros((covariances.shape[0], 6), dtype=means3D.dtype, device="cuda")
    cov3D[:,0] = covariances[:,0,0]
    cov3D[:,1] = covariances[:,0,1]
    cov3D[:,2] = covariances[:,0,2]
    cov3D[:,3] = covariances[:,1,1]
    cov3D[:,4] = covariances[:,1,2]
    cov3D[:,5] = covariances[:,2,2]

    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = values.reshape(-1, 3),
        opacities = opacities.reshape(-1, 1),
        scales = None,
        rotations = None,
        cov3D_precomp = cov3D)

    return rendered_image.transpose(0, -1)

def build_covariances(s, t):
    t = torch.tanh(t)
    S = torch.diag_embed(s)
    T = torch.diag_embed(torch.zeros(s.shape, device="cuda"))
    indices = torch.tril_indices(s.shape[-1], s.shape[-1], -1)

    T[..., indices[0], indices[1]] = t
    T = T + torch.diag_embed(torch.ones(s.shape, device="cuda"))

    return T @ S @ torch.transpose(T, -1, -2)


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
