import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn

import gaussians

from diff_gaussian_sampling import GaussianSampler

res = 64
nx = ny = 10
d = 2

tx = torch.linspace(-1, 1, nx).cuda()
ty = torch.linspace(-1, 1, ny).cuda()
gx, gy = torch.meshgrid((tx,ty), indexing="ij")
means = torch.stack((gx,gy), dim=-1).reshape(nx*ny,d) \
      + (torch.rand((nx*ny,d), device="cuda") * 2.0 - 1.0) * 0.1
scaling = torch.ones((nx*ny,d), device="cuda") * -4.0
scaling = torch.exp(scaling)
transform = torch.zeros((nx*ny,d * (d - 1) // 2), device="cuda")
transform = torch.tanh(transform)

covariances, conics = gaussians.build_covariances(scaling, transform)

values = torch.ones((nx*ny,1), device="cuda") * 0.5

gaussians.plot_gaussians(means, covariances, values)
plt.show()

tx = torch.linspace(-1, 1, res).cuda()
ty = torch.linspace(-1, 1, res).cuda()
gx, gy = torch.meshgrid((tx, ty), indexing="xy")
samples = torch.stack((gx, gy), dim=-1).reshape(res * res, d)

sampler = GaussianSampler(True)
sampler.preprocess(means, values, covariances, conics, samples)
img1 = sampler.sample_gaussians().reshape(res, res).detach().cpu().numpy()

plt.figure()
plt.imshow(img1)
plt.axis("off")
plt.colorbar()
plt.show()
