import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn

import gaussians

from diff_gaussian_sampling import GaussianSampler

n = 20
d = 1

scale = 1.0

means = torch.linspace(-1, 1, n).cuda().reshape(-1, 1) * scale
scaling = torch.ones((n,d), device="cuda") * -5.0

samples = means.unsqueeze(-1) * 4
powers = samples.transpose(-1, -2) @ samples
values = torch.exp(-powers).squeeze(-1)

covariances = torch.exp(scaling)
conics = 1.0 / covariances

res = 200
img_samples = torch.linspace(-1, 1, res).cuda() * scale

sampler = GaussianSampler(True)
sampler.preprocess(means, values, covariances, conics, img_samples)
results = sampler.sample_gaussians().detach().cpu().numpy()

fig = plt.figure()
plt.plot(img_samples.detach().cpu().numpy(), results)
for i in range(n):
    samples = img_samples - means[i]
    powers = -0.5 * conics[i] * samples ** 2
    results = values[i] * torch.exp(powers)
    results = results.detach().cpu().numpy()
    plt.plot(img_samples.detach().cpu().numpy(), results, "--")

plt.show()
plt.close(fig)

results = sampler.sample_gaussians_derivative().detach().squeeze().cpu().numpy()

fig = plt.figure()
plt.plot(img_samples.detach().cpu().numpy(), results)
plt.show()
plt.close(fig)

results = sampler.sample_gaussians_laplacian().detach().squeeze().cpu().numpy()

fig = plt.figure()
plt.plot(img_samples.detach().cpu().numpy(), results)
plt.show()
plt.close(fig)
