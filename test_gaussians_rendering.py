import matplotlib.pyplot as plt
import scipy.stats as st
import torch
import torch.nn.functional as f

from torch import nn

import gaussians

nx = 10
ny = 10
d = 3

tx = torch.linspace(-1, 1, nx).cuda()
ty = torch.linspace(-1, 1, ny).cuda()
gx, gy = torch.meshgrid((tx,ty), indexing="xy")
gz = torch.ones((nx,ny), device="cuda")
means = torch.stack((gx,gy,gz), dim=-1)
scaling = torch.ones((nx,ny,d), device="cuda") * -2.0
transform = torch.zeros((nx,ny, d * (d - 1) // 2), device="cuda")
opacities = torch.ones((nx,ny), device="cuda") * 0.5
conic = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1)

sample_mean = torch.tensor([0.0, 0.0, 0], device="cuda").reshape(1, 1, d, 1)
samples = means.unsqueeze(-1) - sample_mean
powers = -0.5 * (samples.transpose(-1, -2) @ (conic @ samples))
values = torch.exp(powers).repeat(1, 1, d, 1)
values = values / torch.max(values)

scaling = torch.exp(scaling)
transform = f.tanh(transform) * 2.0 - 1.0

covariances = gaussians.build_covariances(scaling, transform)

#gaussians.plot_gaussians(means, covariances, values, opacities,d)

res = 32
img1 = gaussians.sample_gaussians_img(
    means, covariances, opacities, values, res, res, d
).detach().cpu().numpy()[:,:,0]
img2 = gaussians.rasterize_gaussians(
    means, covariances, opacities, values, res, res
).detach().cpu().numpy()[:,:,0]

fig = plt.figure()
ax = fig.subplots(1, 2)
im = ax[0].imshow(img1)
plt.colorbar(im)
im = ax[1].imshow(img2)
plt.colorbar(im)
plt.show()
