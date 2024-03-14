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
nx = ny = 3
d = 2
scale = 2.5

log_step = 100
densification_step = log_step * 1 + 1
warm_up = 0

nu = 1.0 / (100.0 * np.pi)
dt = 1.0
problem = Problem.WAVE

tx = torch.linspace(-1, 1, nx).cuda() * 0.1
ty = torch.linspace(-1, 1, ny).cuda() * 0.1
gx, gy = torch.meshgrid((tx,ty), indexing="ij")
raw_means = torch.atanh(torch.stack((gx,gy), dim=-1).reshape(nx*ny,d))
raw_scaling = torch.ones((nx*ny,d), device="cuda") * -5.0
transform = torch.zeros((nx*ny,d * (d - 1) // 2), device="cuda")

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

sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, d, 1)

sampler = GaussianSampler(True)
torch.autograd.set_detect_anomaly(True)

parameters = [
    { "name": "means", "params": raw_means, "lr": 1e-2 },
    { "name": "values", "params": values, "lr": 1e-2 },
    { "name": "scaling", "params": raw_scaling, "lr": 1e-3 },
    { "name": "transform", "params": transform, "lr": 1e-2 }
]

losses = []
all_losses = []
max_mean_grad = []
max_scale_grad = []

for i in range(10):
    loss = 1.0
    loss_mean = 1.0
    j = 0
    counter = 0

    optim = torch.optim.Adam(parameters)

    mean_grad = torch.zeros_like(raw_means, device="cuda")
    mean_grad_norm = torch.zeros(raw_means.shape[0], device="cuda")
    scale_grad = torch.zeros_like(raw_means, device="cuda")
    scale_grad_norm = torch.zeros(raw_means.shape[0], device="cuda")

    while loss_mean > 1e-4 and j < 5000:
        if problem == Problem.WAVE and i == 0:
            samples = (torch.randn((1024, 2), device="cuda") / 2.0).clamp(-1.0, 1.0) * scale
        else:
            samples = (torch.rand((1024, 2), device="cuda") * 2.0 - 1.0) * scale

        if i > 0:
            with torch.no_grad():
                sampler.preprocess(
                    prev_means, prev_values, prev_covariances, prev_conics, samples)

                prev_img = sampler.sample_gaussians() # n, c
                prev_ux = sampler.sample_gaussians_derivative() # n, d, c
                prev_uxx = sampler.sample_gaussians_laplacian() # n, d, d, c

        means = f.tanh(raw_means) * scale
        scaling = torch.exp(raw_scaling)
        covariances = gaussians.build_covariances(scaling, transform)
        conics = torch.inverse(covariances)

        sampler.preprocess(means, values, covariances, conics, samples)
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
        all_losses.append(loss.item())

        if (j+1) // densification_step > warm_up - 1:
            mean_grad += raw_means.grad
            mean_grad_norm += torch.norm(mean_grad, dim=-1)
            scale_grad += raw_scaling.grad
            scale_grad_norm += torch.norm(scale_grad, dim=-1)
            
            counter += 1

        if ((j+1) % log_step) == 0:
            losses.append(np.mean(all_losses))
            all_losses = []

            loss_tensor = torch.tensor(losses[-min(5, (j+1)//log_step):])
            loss_mean = torch.mean(loss_tensor).item()
            loss_std = torch.std(loss_tensor).item() / loss_mean

            print("Iteration {} - {}".format(i, j))
            print("   loss", loss_mean)
            print("   std", loss_std)

            if loss_std < 0.1:
                for group in optim.param_groups:
                    group["lr"] *= 2.0
            # else:
            #     for group in optim.param_groups:
            #         group["lr"] *= 0.99


        optim.step()
        optim.zero_grad()

        if ((j+1) % densification_step) == 0 and (j+1) // densification_step > warm_up:
            with torch.no_grad():
                mean_grad /= counter
                mean_grad_norm /= counter
                scale_grad /= counter
                scale_grad_norm /= counter
                max_mean_grad.append(torch.amax(mean_grad_norm).item())
                max_scale_grad.append(torch.amax(scale_grad_norm).item())
                counter = 0

                keep_mask = torch.logical_and(
                    torch.norm(values, dim=-1) > 0.01,
                    torch.sum(torch.exp(raw_scaling), dim=-1) < 0.5
                )

                # Roughly 90-th quantile
                quantile = torch.mean(mean_grad_norm) + 1.6 * torch.std(mean_grad_norm)
                split_indices = torch.logical_and(mean_grad_norm > quantile, keep_mask)
                split_len = torch.sum(split_indices).item()
                # split_dir = mean_grad[split_indices]

                extensions = {
                    "means": raw_means.data[split_indices],
                    "values": values.data[split_indices],
                    "scaling": raw_scaling.data[split_indices],
                    "transform": transform.data[split_indices]
                }

                new_tensors = {}

                for group in optim.param_groups:
                    extension = extensions[group["name"]]
                    stored_state = optim.state.get(group["params"][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.cat((
                            stored_state["exp_avg"][keep_mask],
                             torch.zeros_like(extension)
                         ), dim=0)
                        stored_state["exp_avg_sq"] = torch.cat((
                            stored_state["exp_avg_sq"][keep_mask],
                            torch.zeros_like(extension)
                        ), dim=0)

                        del optim.state[group["params"][0]]
                        group["params"][0] = nn.Parameter(torch.cat(
                            (group["params"][0][keep_mask], extension), dim=0).requires_grad_(True))
                        optim.state[group["params"][0]] = stored_state

                        new_tensors[group["name"]] = group["params"][0]
                    else:
                        group["params"][0] = nn.Parameter(torch.cat(
                            (group["params"][0][keep_mask], extension), dim=0).requires_grad_(True))
                        new_tensors[group["name"]] = group["params"][0]

                raw_means = new_tensors["means"]
                values = new_tensors["values"]
                raw_scaling = new_tensors["scaling"]
                transform = new_tensors["transform"]

            print(split_len, torch.sum(~keep_mask).item())

            means = f.tanh(raw_means) * scale
            scaling = torch.exp(raw_scaling)
            covariances = gaussians.build_covariances(scaling, transform)
            conics = torch.inverse(covariances)

            gaussians.plot_gaussians(means, covariances, values[:,1].unsqueeze(-1), scale)
            plt.savefig("results_no_mlp/initialize_gaussians_{}_{}.png".format(i, j), dpi=200)
            plt.close("all")

            mean_grad = torch.zeros_like(raw_means, device="cuda")
            mean_grad_norm = torch.zeros(raw_means.shape[0], device="cuda")
            scale_grad = torch.zeros_like(raw_means, device="cuda")
            scale_grad_norm = torch.zeros(raw_means.shape[0], device="cuda")

        j += 1

    plt.figure()
    plt.plot(np.arange(0, len(losses)*100, 100), losses)
    plt.yscale("log")
    plt.savefig("results_no_mlp/loss_{}.png".format(i))

    plt.figure()
    plt.plot(np.arange(0, len(max_mean_grad)*100, 100), max_mean_grad)
    plt.yscale("log")
    plt.savefig("results_no_mlp/max_mean_grad_{}.png".format(i))

    plt.figure()
    plt.plot(np.arange(0, len(max_mean_grad)*100, 100), max_scale_grad)
    plt.yscale("log")
    plt.savefig("results_no_mlp/max_scale_grad_{}.png".format(i))

    means = f.tanh(raw_means) * scale
    scaling = torch.exp(raw_scaling)
    covariances = gaussians.build_covariances(scaling, transform)
    conics = torch.inverse(covariances)

    prev_means = means.detach().clone()
    prev_covariances = covariances.detach().clone()
    prev_conics = conics.detach().clone()
    prev_values = values.detach().clone()
 
    gaussians.plot_gaussians(means, covariances, values[:,1].unsqueeze(-1), scale)
    plt.savefig("results_no_mlp/initialize_gaussians_{}.png".format(i), dpi=200)

    res = 128

    tx = torch.linspace(-1, 1, res).cuda() * scale
    ty = torch.linspace(-1, 1, res).cuda() * scale
    gx, gy = torch.meshgrid((tx, ty), indexing="xy")
    samples = torch.stack((gx, gy), dim=-1).reshape(res*res, 2)

    with torch.no_grad():
        sampler.preprocess(means, values, covariances, conics, samples)
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
