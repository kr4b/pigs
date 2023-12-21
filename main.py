import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as f

from torch import nn

import gaussians

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.layers(x)

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

model = Network().cuda()
optim = torch.optim.Adam(model.parameters())
model.train()

N = 1000
log_step = N / 100
training_error = []

for epoch in range(N):
    new_means = means.clone()
    new_values = values.clone()
    deltas = torch.zeros((nx, ny, 3), device="cuda")
    for i in range(ny):
        for j in range(nx):
            xyu = torch.tensor([tx[j], ty[i], values[j,i]], device="cuda")
            delta = model(xyu)
            deltas[j,i] = delta / 0.01
            new_means[j,i] += delta[:2]
            new_values[j,i] += delta[2]

    samples = torch.rand((100, 2), device="cuda") * 2.0 - 1.0
    derivatives = gaussians.gaussian_derivative2(
        means, covariances, opacities, values, samples, d) # 100, nx, ny, d
    dx = derivatives.sum(dim=(1,2)) # 100, d

    sample_values = gaussians.sample_gaussians(
        means, covariances, opacities, values, samples, d) # 100, nx, ny, d
    dt = sample_values * deltas[:,:,2].reshape(1, nx, ny, 1)
    dt = dt.sum(dim=(1,2,3)) # 100

    boundaries = torch.cat((-torch.ones(25, device="cuda"), torch.ones(25, device="cuda")))
    bc_samples = torch.zeros((100, 2), device="cuda")
    bc_samples[50:,0] = torch.rand(50, device="cuda") * 2.0 - 1.0
    bc_samples[50:,1] = boundaries
    bc_samples[:50,1] = torch.rand(50, device="cuda") * 2.0 - 1.0
    bc_samples[:50,0] = boundaries
    bc_sample_values = gaussians.sample_gaussians(
        new_means, covariances, opacities, new_values, samples, d) # 100, nx, ny, d

    diffusion_loss = torch.mean((dt - dx.sum(dim=-1)) ** 2)
    bc_loss = torch.mean(bc_sample_values ** 2)
    conservation_loss = torch.mean(deltas[:2].sum(dim=-1) ** 2)
    loss = diffusion_loss + bc_loss + conservation_loss

    loss.backward()
    optim.step()
    optim.zero_grad()

    if epoch % log_step == 0:
        training_error.append(loss.item())
        print("Epoch {}: Total Loss {}".format(epoch, training_error[-1]))
        print("  BC Loss:", bc_loss.item())
        print("  Diffusion Loss:", diffusion_loss.item())
        print("  Conservation Loss:", conservation_loss.item())
        print("  Deltas:", torch.median(deltas[:,:,2]).item(), torch.mean(deltas[:,:,2]).item(), torch.min(deltas[:,:,2]).item(), torch.max(deltas[:,:,2]).item())

plt.figure()
plt.plot(np.arange(0, N, log_step), training_error)
plt.yscale("log")
plt.savefig("training_error.png")

res = 32
img1 = gaussians.sample_gaussians_img(
    means, covariances, opacities, values, res, res, d
).detach().cpu().numpy()

model.eval()
with torch.no_grad():
    new_means = means.clone()
    new_values = values.clone()
    for i in range(ny):
        for j in range(nx):
            xyu = torch.tensor([tx[j], ty[i], values[j,i]], device="cuda")
            delta = model(xyu)
            new_means[j,i] += delta[:2]
            new_values[j,i] += delta[2]

img2 = gaussians.sample_gaussians_img(
    new_means, covariances, opacities, new_values, res, res, d
).detach().cpu().numpy()

img3 = gaussians.sample_gaussians_img(
    means, covariances, opacities, new_values, res, res, d
).detach().cpu().numpy()

img4 = gaussians.sample_gaussians_img(
    new_means, covariances, opacities, values, res, res, d
).detach().cpu().numpy()

fig = plt.figure()
ax = fig.subplots(2, 2)
im = ax[0,0].imshow(img1)
plt.colorbar(im)
im = ax[0,1].imshow(img2)
plt.colorbar(im)
im = ax[1,0].imshow(img3)
plt.colorbar(im)
im = ax[1,1].imshow(img4)
plt.colorbar(im)
plt.savefig("results.png")
