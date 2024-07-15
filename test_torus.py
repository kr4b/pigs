import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn

from diff_gaussian_sampling import GaussianSampler

import gaussians

res = 128
n = 10
d = 2

ty = torch.linspace(-1, 1, n, device="cuda")
means = torch.stack((torch.ones(n, device="cuda") * -0.95, ty), dim=-1)
scaling = torch.ones((n, d), device="cuda") * -3.0
scaling = torch.exp(scaling)
transform = torch.zeros((n, d * (d - 1) // 2), device="cuda")
values = torch.ones((n, 1), device="cuda") * 0.5

covariances, conics = gaussians.build_covariances(scaling, transform)

gaussians.plot_gaussians(means, covariances, values)
plt.show()

tx = torch.linspace(-1, 1, res, device="cuda")
ty = torch.linspace(-1, 1, res, device="cuda")
gx, gy = torch.meshgrid((tx, ty), indexing="xy")
samples = torch.stack((gx, gy), dim=-1).reshape(res*res, 2)

sampler = GaussianSampler(True)
sampler.preprocess(means, values, covariances, conics, samples)
img = sampler.sample_gaussians().reshape(res, res).detach().cpu().numpy()
plt.figure()
plt.imshow(img)
plt.show()
