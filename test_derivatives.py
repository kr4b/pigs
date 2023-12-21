import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as f

from torch import nn

import gaussians
nx = 10
ny = 10
d = 2

tx = torch.linspace(-1, 1, nx).cuda()
ty = torch.linspace(-1, 1, ny).cuda()
gx, gy = torch.meshgrid((tx,ty), indexing="xy")
means = torch.stack((gx,gy), dim=-1)
scaling = torch.ones((nx,ny,d), device="cuda") * -4.0
transform = torch.zeros((nx,ny, d * (d - 1) // 2), device="cuda")
opacities = torch.ones((nx,ny), device="cuda") * 0.5
conic = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1)

sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, 1, d, 1)
samples = means.unsqueeze(-1) - sample_mean
powers = -0.5 * (samples.transpose(-1, -2) @ (conic @ samples))
values = torch.exp(powers)
values = values / torch.max(values)

scaling = torch.exp(scaling)
transform = f.tanh(transform)

covariances = gaussians.build_covariances(scaling, transform)

tx = torch.linspace(-1, 1, 32).cuda()
ty = torch.linspace(-1, 1, 32).cuda()
gx, gy = torch.meshgrid((tx, ty), indexing="xy")
samples = torch.stack((gx, gy), dim=-1).reshape(32 * 32, d)

img = gaussians.gaussian_derivative(means, covariances, opacities, values, samples, d)
img = img.sum(dim=(1,2)).reshape(32, 32, -1).detach().cpu().numpy()

fig = plt.figure()
ax = fig.subplots(1, 2)
ax[0].set_title("x")
im = ax[0].imshow(img[:,:,0])
plt.colorbar(im)
ax[1].set_title("y")
im = ax[1].imshow(img[:,:,1])
plt.colorbar(im)
plt.savefig("derivatives.png")

img = gaussians.gaussian_derivative2(means, covariances, opacities, values, samples, d)
img = img.sum(dim=(1,2)).reshape(32, 32, -1).detach().cpu().numpy()

fig = plt.figure()
ax = fig.subplots(1, 2)
ax[0].set_title("x")
im = ax[0].imshow(img[:,:,0])
plt.colorbar(im)
ax[1].set_title("y")
im = ax[1].imshow(img[:,:,1])
plt.colorbar(im)
plt.savefig("second_derivatives.png")
