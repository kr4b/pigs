import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.io

import imageio.v3 as imageio

from torch import nn

import gaussians

from diff_gaussian_sampling import GaussianSampler

scale = 1.0
d = 2
nu = 1e-3
dt = 1.0

frame0 = torch.load("initialization/V1e-3/f_0-small.pt")
frame1 = torch.load("initialization/V1e-3/f_0-1.pt")

means0 = frame0["means"]
values0 = frame0["values"]
scaling0 = frame0["scaling"]
transforms0 = frame0["transforms"]
covariances0, conics0 = gaussians.build_covariances(scaling0, transforms0)

means1 = frame1["means"]
values1 = frame1["values"]
scaling1 = frame1["scaling"]
transforms1 = frame1["transforms"]
covariances1, conics1 = gaussians.build_covariances(scaling1, transforms1)

res = 128
tx = torch.linspace(-1, 1, res).cuda() * scale
ty = torch.linspace(-1, 1, res).cuda() * scale
gx, gy = torch.meshgrid((tx, ty), indexing="xy")
samples = torch.stack((gx, gy), dim=-1).reshape(res * res, d)

sampler = GaussianSampler(True)

sampler.preprocess(means0, values0, covariances0, conics0, samples)
u0 = sampler.sample_gaussians() # n, c

print(torch.norm(u0, dim=-1).max().item())
exit()

ux0 = sampler.sample_gaussians_derivative() # n, d, c
uxx0 = sampler.sample_gaussians_laplacian() # n, d, d, c
uxxx0 = sampler.sample_gaussians_third_derivative() # n, d, d, d, c

w0 = ux0[:,0,1] - ux0[:,1,0]
wx0 = uxx0[...,0,1] - uxx0[...,1,0]
wxx0 = uxxx0[...,0,1] -  uxxx0[...,1,0]

sampler.preprocess(means1, values1, covariances1, conics1, samples)
u1 = sampler.sample_gaussians() # n, c
ux1 = sampler.sample_gaussians_derivative() # n, d, c
uxx1 = sampler.sample_gaussians_laplacian() # n, d, d, c
uxxx1 = sampler.sample_gaussians_third_derivative() # n, d, d, d, c

w1 = ux1[:,0,1] - ux1[:,1,0]
wx1 = uxx1[...,0,1] - uxx1[...,1,0]
wxx1 = uxxx1[...,0,1] -  uxxx1[...,1,0]

fig = plt.figure()
ax = fig.subplots(1, 2)
ax[0].imshow(w0.reshape(res, res).detach().cpu().numpy())
ax[1].imshow(w1.reshape(res, res).detach().cpu().numpy())
plt.show()

nt = 100
time_samples = torch.arange(1, nt+1, device="cuda").reshape(-1, 1, 1) / nt

u = time_samples * u0.unsqueeze(0) + (1 - time_samples) * u1.unsqueeze(0)
ux = time_samples.reshape(-1, 1, 1, 1) * ux0.unsqueeze(0) \
   + (1 - time_samples.reshape(-1, 1, 1, 1)) * ux1.unsqueeze(0)
uxx = time_samples.reshape(-1, 1, 1, 1, 1) * uxx0.unsqueeze(0) \
    + (1 - time_samples.reshape(-1, 1, 1, 1, 1)) * uxx1.unsqueeze(0)
wx = time_samples * wx0.unsqueeze(0) + (1 - time_samples) * wx1.unsqueeze(0)
wxx = time_samples.reshape(-1, 1, 1, 1) * wxx0.unsqueeze(0) \
    + (1 - time_samples.reshape(-1, 1, 1, 1)) * wxx1.unsqueeze(0)

wt = w1 - w0
rhs = nu * (wxx[...,0,0] + wxx[...,1,1]) - (u[...,0] * wx[...,0] + u[...,1] * wx[...,1])

loss = torch.zeros(res*res, device="cuda")

for i in range(nt):
    loss += (wt - dt * rhs[i]) / nt

print("Average loss:", (loss ** 2).mean().item())

plt.figure()
plt.imshow(loss.reshape(res, res).detach().cpu().numpy())
plt.colorbar()
plt.show()

