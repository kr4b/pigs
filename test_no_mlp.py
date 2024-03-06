import time
import enum
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f

import imageio.v3 as imageio

from torch import nn

import gaussians

from diff_gaussian_sampling import GaussianSampler

class Problem(enum.Enum):
    DIFFUSION = enum.auto()
    POISSON = enum.auto()
    BURGERS = enum.auto()
    WAVE = enum.auto()
    NAVIER_STOKES = enum.auto()

os.makedirs("results_no_mlp", exist_ok=True)

res = 32
nx = ny = 40
d = 2
scale = 2.5

densification_step = 100

nu = 1.0 / (100.0 * np.pi)
dt = 1.0
problem = Problem.WAVE

tx = torch.linspace(-1, 1, nx).cuda() * 0.9
ty = torch.linspace(-1, 1, ny).cuda() * 0.9
gx, gy = torch.meshgrid((tx,ty), indexing="ij")
raw_means = torch.atanh(torch.stack((gx,gy), dim=-1).reshape(nx*ny,d))
raw_scaling = torch.ones((nx*ny,d), device="cuda") * -5.0
transform = torch.zeros((nx*ny,d * (d - 1) // 2), device="cuda")
opacities = torch.ones((nx*ny), device="cuda")

if problem == Problem.WAVE:
    values = torch.zeros((nx*ny,2), device="cuda")
    c = 2
else:
    values = torch.zeros((nx*ny,1), device="cuda")
    c = 1

raw_means = nn.Parameter(raw_means)
values = nn.Parameter(values)
raw_scaling = nn.Parameter(raw_scaling)
transform = nn.Parameter(transform)

keep_mask = torch.ones(nx*ny, dtype=torch.bool, device="cuda")
sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, d, 1)

sampler = GaussianSampler(True)
torch.autograd.set_detect_anomaly(True)

parameters = nn.ParameterList([raw_means, values, raw_scaling, transform])

for i in range(10):
    loss = 1.0
    loss_sum = 0
    j = 0

    optim = torch.optim.Adam(parameters)

    while loss > 1e-4 and j < 1000:
        if problem == Problem.WAVE and i == 0:
            samples = (torch.randn((1024, 2), device="cuda") / 2.0).clamp(-1.0, 1.0) * scale
        else:
            samples = (torch.rand((1024, 2), device="cuda") * 2.0 - 1.0) * scale

        if i > 0:
            with torch.no_grad():
                sampler.preprocess(
                    prev_means, prev_values, prev_covariances, prev_conics, opacities, samples)

                prev_img = sampler.sample_gaussians() # n, c
                prev_ux = sampler.sample_gaussians_derivative() # n, d, c
                prev_uxx = sampler.sample_gaussians_laplacian() # n, d, d, c

        means = f.tanh(raw_means)[keep_mask] * scale
        scaling = torch.exp(raw_scaling)[keep_mask]
        covariances = gaussians.build_covariances(scaling, transform[keep_mask])
        conics = torch.inverse(covariances)

        sampler.preprocess(means, values[keep_mask], covariances, conics, opacities, samples)
        img = sampler.sample_gaussians() # n, c

        if i == 0:
            _samples = samples.unsqueeze(-1) - sample_mean
            if problem == Problem.WAVE:
                _conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.01 * scale)
            else:
                _conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1 * scale)
            powers = -0.5 * (_samples.transpose(-1, -2) @ (_conics @ _samples))
            desired = torch.exp(powers).squeeze() * 1.5

            if problem == Problem.WAVE:
                loss = torch.mean((img[...,1] - desired) ** 2)
                loss += torch.mean(img[...,0] ** 2)
            else:
                loss = torch.mean((img[...,0] - desired) ** 2)
        else:
            time_samples = torch.rand((1024), device="cuda")

            ux = sampler.sample_gaussians_derivative() # n, d, c
            uxx = sampler.sample_gaussians_laplacian() # n, d, d, c

            ut = (img - prev_img) / dt
            #u = img#(img + prev_img) / 2.0
            u = time_samples.reshape(-1,1) * prev_img + (1 - time_samples.reshape(-1,1)) * img
            #ux = ux#(ux + prev_ux) / 2.0
            ux = time_samples.reshape(-1,1,1) * prev_ux + (1 - time_samples.reshape(-1,1,1)) * ux
            #uxx = uxx#(uxx + prev_uxx) / 2.0
            uxx = time_samples.reshape(-1,1,1,1) * prev_uxx + (1 - time_samples.reshape(-1,1,1,1)) * uxx

            if problem == Problem.WAVE:
                loss1 = torch.mean(
                    (ut[:,1] - (10 * (uxx[:,0,0,0] + uxx[:,1,1,0]) - 0.1 * u[:,1])) ** 2)
                loss2 = torch.mean((ut[:,0] - u[:,1]) ** 2)
                loss = 0.01 * loss1 + loss2
            elif problem == Problem.BURGERS:
                loss = torch.mean(
                    (ut[:,0] - (nu * (uxx[:,0,0] + uxx[:,1,1]) - u[:,0] * ux[:,0])) ** 2)
            elif problem == Problem.DIFFUSION:
                loss = torch.mean((ut[:,0] - (uxx[:,0,0] + uxx[:,1,1])) ** 2)

        loss.backward()
        optim.step()
        optim.zero_grad()

        loss_sum += loss.item()

        if ((j+1) % densification_step) == 0:
            print("Iteration {} - {}".format(i, j))
            print("   loss", loss_sum / densification_step)
            loss_sum = 0

            res = 64

            tx = torch.linspace(-1, 1, res).cuda() * scale
            ty = torch.linspace(-1, 1, res).cuda() * scale
            gx, gy = torch.meshgrid((tx, ty), indexing="xy")
            samples = torch.stack((gx, gy), dim=-1).reshape(res*res, 2)

            with torch.no_grad():
                sampler.preprocess(means, values[keep_mask], covariances, conics, opacities, samples)
                img = sampler.sample_gaussians().reshape(res,res,c)

            samples = samples.unsqueeze(-1) - sample_mean
            if problem == Problem.WAVE:
                conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.01 * scale)
            else:
                conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1 * scale)
            powers = -0.5 * (samples.transpose(-1, -2) @ (conics @ samples))
            desired = torch.exp(powers).reshape(res,res)

            fig = plt.figure(figsize=(9,4))
            ax = fig.subplots(1, 2)
            if problem == Problem.WAVE:
                vmin = torch.min(img)
                vmax = torch.max(img)

                im = ax[0].imshow(img[...,0].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
                im = ax[1].imshow(img[...,1].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
            else:
                vmin = min(torch.min(desired), torch.min(img))
                vmax = max(torch.max(desired), torch.max(img))

                im = ax[0].imshow(img[...,0].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
                im = ax[1].imshow(desired.detach().cpu().numpy(), vmin=vmin, vmax=vmax)

            ax[0].axis("off")
            ax[0].invert_yaxis()
            ax[1].axis("off")
            ax[1].invert_yaxis()
            cbar_ax = fig.add_axes([0.925, 0.1, 0.025, 0.8])
            fig.colorbar(im, cax=cbar_ax)
            plt.savefig("results_no_mlp/initialize_{}_{}.png".format(i, j))
            plt.close(fig)

        j += 1

    means = f.tanh(raw_means)[keep_mask] * scale
    scaling = torch.exp(raw_scaling)[keep_mask]
    covariances = gaussians.build_covariances(scaling, transform[keep_mask])
    conics = torch.inverse(covariances)

    prev_means = means.detach().clone()
    prev_covariances = covariances.detach().clone()
    prev_conics = conics.detach().clone()
    prev_values = values[keep_mask].detach().clone()

    gaussians.plot_gaussians(
        means.detach(), covariances, opacities * 0.25, values[keep_mask].detach(), scale)
    plt.savefig("results_no_mlp/initialize_gaussians_{}.png".format(i), dpi=200)

    res = 128

    tx = torch.linspace(-1, 1, res).cuda() * scale
    ty = torch.linspace(-1, 1, res).cuda() * scale
    gx, gy = torch.meshgrid((tx, ty), indexing="xy")
    samples = torch.stack((gx, gy), dim=-1).reshape(res*res, 2)

    with torch.no_grad():
        sampler.preprocess(means, values[keep_mask], covariances, conics, opacities, samples)
        img = sampler.sample_gaussians().reshape(res,res,c)

    samples = samples.unsqueeze(-1) - sample_mean
    if problem == Problem.WAVE:
        conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.01 * scale)
    else:
        conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1 * scale)
    powers = -0.5 * (samples.transpose(-1, -2) @ (conics @ samples))
    desired = torch.exp(powers).reshape(res,res)


    fig = plt.figure(figsize=(9,4))
    ax = fig.subplots(1, 2)
    if problem == Problem.WAVE:
        vmin = torch.min(img)
        vmax = torch.max(img)

        im = ax[0].imshow(img[...,0].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
        im = ax[1].imshow(img[...,1].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
    else:
        vmin = min(torch.min(desired), torch.min(img))
        vmax = max(torch.max(desired), torch.max(img))

        im = ax[0].imshow(img[...,0].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
        im = ax[1].imshow(desired.detach().cpu().numpy(), vmin=vmin, vmax=vmax)

    ax[0].axis("off")
    ax[0].invert_yaxis()
    ax[1].axis("off")
    ax[1].invert_yaxis()
    cbar_ax = fig.add_axes([0.925, 0.1, 0.025, 0.8])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig("results_no_mlp/initialize_{}.png".format(i))
    plt.close(fig)
