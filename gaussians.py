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

def plot_gaussians(means, covariances, values, opacities, d):
    nx, ny = covariances.shape[:2]

    means = means.cpu().numpy()
    values = values.cpu().numpy()
    opacities = opacities.cpu().numpy()
    covariances = covariances.cpu().numpy().reshape(-1,d,d)
    covariance = np.zeros((covariances.shape[0], 3, 3))
    covariance[:,:d,:d] = covariances
    covariance[:,2,2] = 1.0

    plt.figure()
    ax = plt.gca()

    for i in range(ny):
        for j in range(nx):
            color = matplotlib.cm.get_cmap('viridis')
            ellipse = Ellipse(xy=(0.0, 0.0), width=25.0, height=25.0, fc=color(values[i,j]), alpha=opacities[i,j].item())
            affine = Affine2D(covariance[i*nx+j,:,:]).translate(*means[i,j,:2])
            ellipse.set_transform(affine + ax.transData)

            ax.add_patch(ellipse)

    plt.axis("scaled")


# def compute_gaussian(all_densities, opacities, values, nx, ny, k, l, d):
#     value = 0
#     densities = []
#     for i in range(ny):
#         for j in range(nx):
#             density = all_densities[k,l,j,i]
#             if density > 0.1:
#                 densities.append((density, (j,i)))
# 
#     densities = sorted(densities, key=lambda d: d[0], reverse=True)
#     opacity = 1
#     for (density, (j,i)) in densities:
#         alpha = density * opacities[j,i]
#         test_opacity = opacity * (1 - alpha)
#         if test_opacity < 1e-4:
#             break
#         value += alpha * opacity * values[j,i]
#         opacity = test_opacity
# 
#     return value

# def render_gaussians():
#     img = np.zeros((h, w))
#     for k in range(w):
#        for l in range(h):
#            img[l,k] = compute_gaussian(densities, opacities, values, nx, ny, k, l)[0]

def sample_gaussians(means, covariances, opacities, values, samples, d):
    nx, ny = covariances.shape[:2]

    x = samples.reshape(-1, 1, 1, d, 1) - means.reshape(1, nx, ny, d, 1)
    conics = torch.inverse(covariances)
    powers = -0.5 * (x.transpose(-1, -2) @ (conics @ x))
    densities = torch.exp(powers).reshape(-1, nx, ny, 1)

    opacities = opacities.reshape(1, nx, ny, 1)
    values = values.reshape(1, nx, ny, -1)

    res = densities * opacities * values
    res = res / densities.sum(dim=(1,2)).reshape(-1, 1, 1, 1)

    return res

def _sample_gaussians_region(d, center, size, dx):
    t = []
    for i in range(d):
        half_size = (size[i] - 1) / 2
        t.append(center[i] + torch.linspace(-half_size * dx[i], half_size * dx[i], size[i]))

    grid = torch.meshgrid(t, indexing="xy")
    samples = torch.stack(grid, dim=-1).reshape(-1, d)

    return samples

def sample_gaussians_region(means, covariances, opacities, values, d, center, size, dx):
    samples = _sample_gaussians_region(d, center, size, dx)
    return sample_gaussians(means, covariances, opacities, values, samples, d)

def sample_gaussians_img(means, covariances, opacities, values, w, h, d):
    tx = torch.linspace(-1, 1, w).cuda()
    ty = torch.linspace(-1, 1, h).cuda()
    gx, gy = torch.meshgrid((tx, ty), indexing="xy")
    if d == 3:
        gz = torch.ones((w,h), device="cuda")
        samples = torch.stack((gx, gy, gz), dim=-1).reshape(w * h, d)
    if d == 2:
        samples = torch.stack((gx, gy), dim=-1).reshape(w * h, d)

    img = sample_gaussians(means, covariances, opacities, values, samples, d)
    img = img.sum(dim=(1,2))

    return img.reshape(w, h, -1)

def gaussian_derivative(means, covariances, opacities, values, samples, d):
    nx, ny = covariances.shape[:2]

    x = samples.reshape(-1, 1, 1, d, 1) - means.reshape(1, nx, ny, d, 1)
    conics = torch.inverse(covariances)
    powers = -0.5 * (x.transpose(-1, -2) @ (conics @ x))
    densities = torch.exp(powers).reshape(-1, nx, ny, 1)
    derivatives = -(conics @ x).reshape(-1, nx, ny, d) * densities

    opacities = opacities.reshape(1, nx, ny, 1)
    values = values.reshape(1, nx, ny, -1)

    res = derivatives * opacities * values
    res = res / densities.sum(dim=(1,2)).reshape(-1, 1, 1, 1)

    return res

def gaussian_derivative2(means, covariances, opacities, values, samples, d):
    nx, ny = covariances.shape[:2]

    x = samples.reshape(-1, 1, 1, d, 1) - means.reshape(1, nx, ny, d, 1)
    conics = torch.inverse(covariances)
    powers = -0.5 * (x.transpose(-1, -2) @ (conics @ x))
    densities = torch.exp(powers).reshape(-1, nx, ny, 1)
    ones = torch.ones(x.shape, device="cuda")
    derivatives = ((conics @ x) ** 2  - (conics @ ones)).reshape(-1, nx, ny, d) * densities

    opacities = opacities.reshape(1, nx, ny, 1)
    values = values.reshape(1, nx, ny, -1)

    res = derivatives * opacities * values
    res = res / densities.sum(dim=(1,2)).reshape(-1, 1, 1, 1)

    return res

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
                anyTrue |= torch.all(torch.isclose(sample, element))

            self.assertTrue(anyTrue)

    def test_sample_region_2d_2(self):
        samples = _sample_gaussians_region(2, torch.zeros(2), (2, 2), torch.ones(2))
        self.assertEqual(samples.shape, (4, 2))
        expected = [
            torch.tensor([-0.5, -0.5]),
            torch.tensor([0.5, -0.5]),
            torch.tensor([-0.5, 0.5]),
            torch.tensor([0.5, 0.5]),
        ]
        self.contains(expected, samples)

    def test_sample_region_2d_3(self):
        samples = _sample_gaussians_region(2, torch.tensor([0.0, 0.5]), (3, 3), torch.ones(2) * 0.5)
        self.assertEqual(samples.shape, (9, 2))
        expected = [
            torch.tensor([-0.5, 0.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.5, 0.0]),
            torch.tensor([-0.5, 0.5]),
            torch.tensor([0.0, 0.5]),
            torch.tensor([0.5, 0.5]),
            torch.tensor([-0.5, 1.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.5, 1.0]),
        ]
        self.contains(expected, samples)

    def test_sample_region_3d_2(self):
        samples = _sample_gaussians_region(3, torch.tensor([2.0, -1.0, 0.0]), (2, 2, 2), torch.ones(3) * 2.0)
        self.assertEqual(samples.shape, (8, 3))
        expected = [
            torch.tensor([1.0, -2.0, -1.0]),
            torch.tensor([3.0, -2.0, -1.0]),
            torch.tensor([1.0, 0.0, -1.0]),
            torch.tensor([3.0, 0.0, -1.0]),
            torch.tensor([1.0, -2.0, 1.0]),
            torch.tensor([3.0, -2.0, 1.0]),
            torch.tensor([1.0, 0.0, 1.0]),
            torch.tensor([3.0, 0.0, 1.0]),
        ]
        self.contains(expected, samples)

if __name__ == '__main__':
    unittest.main()
