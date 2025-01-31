import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn

import gaussians

from diff_gaussian_sampling import GaussianSampler

nx = 10
ny = 10
d = 2

tx = torch.linspace(-1, 1, nx).cuda()
ty = torch.linspace(-1, 1, ny).cuda()
gx, gy = torch.meshgrid((tx,ty), indexing="ij")
means = torch.stack((gx,gy), dim=-1).reshape(nx*ny, d)
scaling = torch.ones((nx*ny,d), device="cuda") * -4.0# + 0.5 * torch.linspace(0, 1, nx).cuda().reshape(nx, 1, 1).repeat(1, ny, d)
transform = torch.zeros((nx*ny, d * (d - 1) // 2), device="cuda")
# for i in range(ny):
#     for j in range(nx):
#         transform[j,i,0] = 0.5 + i * j * 0.01
conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1)

sample_mean = torch.tensor([0.1, 0.4], device="cuda").reshape(1, d, 1)
samples = means.unsqueeze(-1) - sample_mean
powers = -0.5 * (samples.transpose(-1, -2) @ (conics @ samples))
values = torch.exp(powers).squeeze(-1)

scaling = torch.exp(scaling)
transform = torch.tanh(transform)

full_covariances, full_conics = gaussians.build_full_covariances(scaling, transform)
covariances, conics = gaussians.flatten_covariances(full_covariances, full_conics)

gaussians.plot_gaussians(means, covariances, values)
plt.show()

res = 256
tx = torch.linspace(-1, 1, res).cuda()
ty = torch.linspace(-1, 1, res).cuda()
gx, gy = torch.meshgrid((tx, ty), indexing="xy")
samples = torch.stack((gx, gy), dim=-1).reshape(res * res, d)

start = time.time()
result_py = gaussians.sample_gaussians(means, full_conics, values, samples)
img_py = result_py.reshape(res, res).detach().cpu().numpy()
print("Original:", time.time() - start)

sampler = GaussianSampler(True)

start = time.time()
sampler.preprocess(means, values, covariances, conics, samples)
result_cuda = sampler.sample_gaussians()
img_cuda = result_cuda.reshape(res, res).detach().cpu().numpy()
print("CUDA:", time.time() - start)

fig = plt.figure(figsize=(10,4))
ax = fig.subplots(1, 3)
ax[0].set_title("Original")
im = ax[0].imshow(img_py)
ax[0].axis("off")
plt.colorbar(im)
ax[1].set_title("CUDA")
im = ax[1].imshow(img_cuda)
ax[1].axis("off")
plt.colorbar(im)
ax[2].set_title("Error")
im = ax[2].imshow(img_py - img_cuda)
ax[2].axis("off")
plt.colorbar(im)
plt.tight_layout()
plt.show()
