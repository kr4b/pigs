import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn

import gaussians

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
opacities = torch.ones((nx*ny), device="cuda") * 0.25

covariances = gaussians.build_covariances(scaling, transform)
conics = torch.inverse(covariances)

values = torch.ones((nx*ny,1), device="cuda") * 0.5

gaussians.plot_gaussians(means, covariances, opacities, values)
plt.savefig("density_gaussians.png")

img1 = gaussians.sample_gaussians_img(
    means, conics, opacities, values, res, res, 1.0).detach().cpu().numpy()

plt.figure()
plt.imshow(img1)
plt.axis("off")
plt.colorbar()
plt.savefig("density.png")
