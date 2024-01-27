import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f

from torch import nn

import gaussians

res = 64
nx = ny = 10
d = 2

tx = torch.linspace(-1, 1, nx).cuda()
ty = torch.linspace(-1, 1, ny).cuda()
gx, gy = torch.meshgrid((tx,ty), indexing="ij")
means = torch.stack((gx,gy), dim=-1)
scaling = torch.ones((nx,ny,d), device="cuda") * -4.0
scaling = torch.exp(scaling)
transform = torch.zeros((nx,ny,d * (d - 1) // 2), device="cuda")
transform = f.tanh(transform)
opacities = torch.ones((nx,ny), device="cuda") * 0.25

covariances = gaussians.build_covariances(scaling, transform)
conics = torch.inverse(covariances)

sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, 1, d, 1)
samples = means.unsqueeze(-1) - sample_mean
conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1)
powers = -0.5 * (samples.transpose(-1, -2) @ (conics @ samples))
u = torch.exp(powers).squeeze(-1)
u = u / torch.max(u)

gaussians.plot_gaussians(means, covariances, opacities, u)
plt.savefig("plot_gaussians1.png")

img1 = gaussians.sample_gaussians_img(
    means, conics, opacities, u, res, res).detach().cpu().numpy()

random_opacities = torch.rand((nx,ny), device="cuda") * 0.8 + 0.1

gaussians.plot_gaussians(means, covariances, random_opacities, u)
plt.savefig("plot_gaussians2.png")

img2 = gaussians.sample_gaussians_img(
    means, conics, random_opacities, u, res, res).detach().cpu().numpy()

random_means = torch.stack((gx,gy), dim=-1) + (torch.rand((nx, ny, d), device="cuda") * 2.0 - 1.0) * 0.1

random_covariances = gaussians.build_covariances(scaling, transform)
random_conics = torch.inverse(random_covariances)

sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, 1, d, 1)
samples = random_means.unsqueeze(-1) - sample_mean
random_conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1)
powers = -0.5 * (samples.transpose(-1, -2) @ (random_conics @ samples))
random_u = torch.exp(powers).squeeze(-1)
random_u = random_u / torch.max(random_u)

gaussians.plot_gaussians(random_means, random_covariances, opacities, random_u)
plt.savefig("plot_gaussians3.png")

img3 = gaussians.sample_gaussians_img(
    random_means, random_conics, opacities, random_u, res, res).detach().cpu().numpy()

scaling = (torch.rand((nx,ny,d), device="cuda") * 2.0 - 1.0) * 0.5 - 4.0
scaling = torch.exp(scaling)

random_covariances = gaussians.build_covariances(scaling, transform)
random_conics = torch.inverse(random_covariances)

gaussians.plot_gaussians(random_means, random_covariances, opacities, random_u)
plt.savefig("plot_gaussians4.png")

img4 = gaussians.sample_gaussians_img(
    random_means, random_conics, opacities, random_u, res, res).detach().cpu().numpy()

transform = (torch.rand((nx,ny,d * (d - 1) // 2), device="cuda") * 2.0 - 1.0)
transform = f.tanh(transform)

random_covariances = gaussians.build_covariances(scaling, transform)
random_conics = torch.inverse(random_covariances)

gaussians.plot_gaussians(random_means, random_covariances, opacities, random_u)
plt.savefig("plot_gaussians5.png")

img5 = gaussians.sample_gaussians_img(
    random_means, random_conics, opacities, random_u, res, res).detach().cpu().numpy()

gaussians.plot_gaussians(means, random_covariances, opacities, u)
plt.savefig("plot_gaussians6.png")

img6 = gaussians.sample_gaussians_img(
    means, random_conics, opacities, u, res, res).detach().cpu().numpy()

vmin = min(np.min(img1), np.min(img2), np.min(img3), np.min(img4), np.min(img5), np.min(img6))
vmax = max(np.max(img1), np.max(img2), np.max(img3), np.max(img4), np.max(img5), np.max(img6))

fig = plt.figure()
ax = fig.subplots(3, 2)
im = ax[0,0].imshow(img1, vmin=vmin, vmax=vmax)
ax[0,0].set_title("Uniform")
im = ax[0,1].imshow(img2, vmin=vmin, vmax=vmax)
ax[0,1].set_title("Random O")
im = ax[1,0].imshow(img3, vmin=vmin, vmax=vmax)
ax[1,0].set_title("Random M")
im = ax[1,1].imshow(img4, vmin=vmin, vmax=vmax)
ax[1,1].set_title("Random MS")
im = ax[2,0].imshow(img5, vmin=vmin, vmax=vmax)
ax[2,0].set_title("Random MSC")
im = ax[2,1].imshow(img6, vmin=vmin, vmax=vmax)
ax[2,1].set_title("Random SC")
cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
fig.colorbar(im, cax=cbar_ax)
plt.tight_layout()
plt.savefig("sampled_gaussians.png", bbox_inches="tight", dpi=200)
plt.close(fig)

diff1 = img1 - img1
diff2 = img2 - img1
diff3 = img3 - img1
diff4 = img4 - img1
diff5 = img5 - img1
diff6 = img6 - img1

vmin = min(np.min(diff1), np.min(diff2), np.min(diff3), np.min(diff4), np.min(diff5), np.min(diff6))
vmax = max(np.max(diff1), np.max(diff2), np.max(diff3), np.max(diff4), np.max(diff5), np.max(diff6))

fig = plt.figure()
ax = fig.subplots(3, 2)
im = ax[0,0].imshow(diff1, vmin=vmin, vmax=vmax)
ax[0,0].set_title("Uniform")
im = ax[0,1].imshow(diff2, vmin=vmin, vmax=vmax)
ax[0,1].set_title("Random O")
im = ax[1,0].imshow(diff3, vmin=vmin, vmax=vmax)
ax[1,0].set_title("Random M")
im = ax[1,1].imshow(diff4, vmin=vmin, vmax=vmax)
ax[1,1].set_title("Random MS")
im = ax[2,0].imshow(diff5, vmin=vmin, vmax=vmax)
ax[2,0].set_title("Random MSC")
im = ax[2,1].imshow(diff6, vmin=vmin, vmax=vmax)
ax[2,1].set_title("Random SC")
cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
fig.colorbar(im, cax=cbar_ax)
plt.tight_layout()
plt.savefig("sampled_gaussians_error.png", bbox_inches="tight", dpi=200)
plt.close(fig)
