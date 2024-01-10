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
for i in range(ny):
    for j in range(nx):
        transform[j,i,0] = 0.5
opacities = torch.ones((nx,ny), device="cuda") * 0.5
conic = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1)

sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, 1, d, 1)
samples = means.unsqueeze(-1) - sample_mean
powers = -0.5 * (samples.transpose(-1, -2) @ (conic @ samples))
values = torch.exp(powers).squeeze(-1)
values = values / torch.max(values)

scaling = torch.exp(scaling)
transform = f.tanh(transform)

covariances = gaussians.build_covariances(scaling, transform)
conics = torch.inverse(covariances)
inv_sqrt_det = torch.sqrt(torch.det(conics))

res = 64
tx = torch.linspace(-1, 1, res).cuda()
ty = torch.linspace(-1, 1, res).cuda()
gx, gy = torch.meshgrid((tx, ty), indexing="xy")
samples = torch.stack((gx, gy), dim=-1).reshape(res * res, d)
samples.requires_grad = True

result = gaussians.sample_gaussians(means, inv_sqrt_det, conics, opacities, values, samples)
grad1 = torch.autograd.grad(result.sum(), samples, retain_graph=True, create_graph=True)[0]

img = gaussians.gaussian_derivative(means, inv_sqrt_det, conics, opacities, values, samples)
img1 = img.sum(dim=(1,2)).reshape(res, res, -1).detach().cpu().numpy()
img2 = grad1.reshape(res, res, -1).detach().cpu().numpy()
img3 = img1-img2

fig = plt.figure()
plt.imshow(result.sum(dim=(1,2)).reshape(res, res).detach().cpu().numpy())
plt.colorbar()
plt.savefig("input.png")

gaussians.plot_gaussians(means, covariances, opacities, values)
plt.savefig("gaussian_plot.png")

fig = plt.figure()
ax = fig.subplots(1, 2)
ax[0].set_title("x")
im = ax[0].imshow(img1[:,:,0])
plt.colorbar(im)
ax[1].set_title("y")
im = ax[1].imshow(img1[:,:,1])
plt.colorbar(im)
plt.savefig("derivatives.png")

fig = plt.figure()
ax = fig.subplots(1, 2)
ax[0].set_title("x")
im = ax[0].imshow(img2[:,:,0])
plt.colorbar(im)
ax[1].set_title("y")
im = ax[1].imshow(img2[:,:,1])
plt.colorbar(im)
plt.savefig("derivatives_autograd.png")

fig = plt.figure()
ax = fig.subplots(1, 2)
ax[0].set_title("x")
im = ax[0].imshow(img3[:,:,0])
plt.colorbar(im)
ax[1].set_title("y")
im = ax[1].imshow(img3[:,:,1])
plt.colorbar(im)
plt.savefig("derivatives_error.png")

grad2_1 = torch.autograd.grad(grad1[:,0].sum(), samples, retain_graph=True)[0]
grad2_2 = torch.autograd.grad(grad1[:,1].sum(), samples)[0]
grad2 = torch.cat((grad2_1, grad2_2), dim=-1)

img = gaussians.gaussian_derivative2(means, inv_sqrt_det, conics, opacities, values, samples)
img1 = img.sum(dim=(1,2)).reshape(res, res, -1).detach().cpu().numpy()
img2 = grad2.reshape(res, res, -1).detach().cpu().numpy()
img3 = img1-img2

fig = plt.figure()
ax = fig.subplots(2, 2)
ax[0,0].set_title("xx")
im = ax[0,0].imshow(img1[:,:,0])
plt.colorbar(im)
ax[0,1].set_title("xy")
im = ax[0,1].imshow(img1[:,:,1])
plt.colorbar(im)
ax[1,0].set_title("yx")
im = ax[1,0].imshow(img1[:,:,2])
plt.colorbar(im)
ax[1,1].set_title("yy")
im = ax[1,1].imshow(img1[:,:,3])
plt.colorbar(im)
plt.savefig("second_derivatives.png")

fig = plt.figure()
ax = fig.subplots(2, 2)
ax[0,0].set_title("xx")
im = ax[0,0].imshow(img2[:,:,0])
plt.colorbar(im)
ax[0,1].set_title("xy")
im = ax[0,1].imshow(img2[:,:,1])
plt.colorbar(im)
ax[1,0].set_title("yx")
im = ax[1,0].imshow(img2[:,:,2])
plt.colorbar(im)
ax[1,1].set_title("yy")
im = ax[1,1].imshow(img2[:,:,3])
plt.colorbar(im)
plt.savefig("second_derivatives_autograd.png")

fig = plt.figure()
ax = fig.subplots(2, 2)
ax[0,0].set_title("xx")
im = ax[0,0].imshow(img3[:,:,0])
plt.colorbar(im)
ax[0,1].set_title("xy")
im = ax[0,1].imshow(img3[:,:,1])
plt.colorbar(im)
ax[1,0].set_title("yx")
im = ax[1,0].imshow(img3[:,:,2])
plt.colorbar(im)
ax[1,1].set_title("yy")
im = ax[1,1].imshow(img3[:,:,3])
plt.colorbar(im)
plt.savefig("second_derivatives_error.png")
