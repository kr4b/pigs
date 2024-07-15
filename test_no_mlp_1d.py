import time
import enum
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

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

os.makedirs("results_no_mlp_1d", exist_ok=True)

n = 25
d = 1
scale = 2.5

n_samples = 128

log_step = 100
densification_step = log_step * 50 + 1
warm_up = 100

nu = 1.0 / (100.0 * np.pi)
dt = 0.05
problem = Problem.BURGERS

raw_means = torch.linspace(-1, 1, n).cuda().reshape(-1, d)
scaling = torch.ones((n,d), device="cuda") * -4.0

if problem == Problem.WAVE:
    values = torch.zeros((n,2), device="cuda")
    c = 2
else:
    values = torch.zeros((n,1), device="cuda")
    c = 1

raw_means = nn.Parameter(raw_means)
values = nn.Parameter(values)
scaling = nn.Parameter(scaling)

sampler = GaussianSampler(True)
torch.autograd.set_detect_anomaly(True)

parameters = [
    { "name": "means", "params": raw_means, "lr": 1e-2 },
    { "name": "values", "params": values, "lr": 1e-2 },
    { "name": "scaling", "params": scaling, "lr": 1e-2 },
]

losses = []
all_losses = []
max_mean_grad = []
max_scale_grad = []

sample_mean = 0.0

res = 200
img_samples = torch.linspace(-1, 1, res).cuda() * scale

means = torch.tanh(raw_means) * scale
covariances = torch.exp(scaling).reshape(-1, d, d)
conics = 1.0 / covariances

# gt = np.load("burgers_gt.npy")

for i in range(13):
    loss = 1.0
    loss_mean = 1.0
    j = 0
    counter = 0

    optim = torch.optim.Adam(parameters)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)

    mean_grad = torch.zeros_like(raw_means, device="cuda")
    mean_grad_norm = torch.zeros(raw_means.shape[0], device="cuda")
    scale_grad = torch.zeros_like(raw_means, device="cuda")
    scale_grad_norm = torch.zeros(raw_means.shape[0], device="cuda")

    if i > 7:
        for group in optim.param_groups:
            group["lr"] = 1e-4

    while loss_mean > 1e-4 and j < 7000:
        samples = (torch.rand((n_samples, d), device="cuda") * 2.0 - 1.0) * scale

        if i > 0:
            with torch.no_grad():
                sampler.preprocess(
                    prev_means, prev_values, prev_covariances, prev_conics, samples)

                prev_img = sampler.sample_gaussians() # n, c
                prev_ux = sampler.sample_gaussians_derivative() # n, d, c
                prev_uxx = sampler.sample_gaussians_laplacian() # n, d, d, c

        means = torch.tanh(raw_means) * scale
        covariances = torch.exp(scaling).reshape(-1, 1)
        conics = 1.0 / covariances

        sampler.preprocess(means, values, covariances, conics, samples)
        img = sampler.sample_gaussians() # n, c

        if i == 0:
            _conics = torch.inverse(
                torch.diag(torch.ones(d, device="cuda")) * np.exp(-4.0) * scale)

            _samples = samples - sample_mean
            # powers = -0.5 * (_samples * _conics * _samples)
            powers = -2.0 * _samples ** 2
            desired = torch.exp(powers).squeeze()

            if problem == Problem.WAVE:
                loss = torch.mean((img[...,0] - desired) ** 2)
                loss += torch.mean((img[...,1] - desired) ** 2)
            else:
                loss = torch.mean((img[...,0] - desired) ** 2)
        else:
            time_samples = torch.rand((n_samples), device="cuda")

            ux = sampler.sample_gaussians_derivative() # n, d, c
            uxx = sampler.sample_gaussians_laplacian() # n, d, d, c

            ut = (img - prev_img) / dt
            #u = img#(img + prev_img) / 2.0
            u = time_samples.reshape(-1,1) * prev_img + (1 - time_samples.reshape(-1,1)) * img
            #ux = ux#(ux + prev_ux) / 2.0
            ux = time_samples.reshape(-1,1,1) * prev_ux + (1 - time_samples.reshape(-1,1,1)) * ux
            #uxx = uxx#(uxx + prev_uxx) / 2.0
            uxx = time_samples.reshape(-1,1,1,1)*prev_uxx + (1 - time_samples.reshape(-1,1,1,1))*uxx

            if problem == Problem.WAVE:
                loss1 = torch.mean((ut[:,1] - (10 * uxx[:,0,0,0] - 0.1 * u[:,1])) ** 2)
                loss2 = torch.mean((ut[:,0] - u[:,1]) ** 2)
                loss = 0.1 * loss1 + loss2
            elif problem == Problem.BURGERS:
                loss = torch.mean((ut[:,0] - (nu * uxx[:,0,0,0] - u[:,0] * ux[:,0,0])) ** 2)
            elif problem == Problem.DIFFUSION:
                loss = torch.mean((ut[:,0] - uxx[:,0,0,0]) ** 2)

        loss.backward()
        all_losses.append(loss.item())

        if (j+1) // densification_step > warm_up - 1:
            mean_grad += raw_means.grad
            mean_grad_norm += torch.norm(mean_grad, dim=-1)
            scale_grad += scaling.grad
            scale_grad_norm += torch.norm(scale_grad, dim=-1)
            
            counter += 1

        if ((j+1) % log_step) == 0:
            scheduler.step()
            losses.append(np.mean(all_losses))
            all_losses = []

            loss_tensor = torch.tensor(losses[-min(5, (j+1)//log_step):])
            loss_mean = torch.mean(loss_tensor).item()
            loss_std = torch.std(loss_tensor).item() / loss_mean

            print("Iteration {} - {}".format(i, j))
            print("   loss", loss_mean)
            # print("   std", loss_std)
            for group in optim.param_groups:
                group["lr"] = max(group["lr"], 1e-5)
            for group in optim.param_groups:
                print("   lr", group["lr"])
                break

            # if loss_std < 0.1:
            #     for group in optim.param_groups:
            #         group["lr"] *= 2.0
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
                print("   Mean grad", max_mean_grad[-1])
                print("   Scale grad", max_scale_grad[-1])
                counter = 0

                keep_mask = torch.logical_and(
                    torch.norm(values, dim=-1) > 0.01,
                    torch.logical_and(
                        torch.sum(torch.abs(raw_means), dim=-1) < 10.0,
                        torch.logical_and(
                            torch.sum(scaling, dim=-1) < 0.0,
                            torch.sum(scaling, dim=-1) > -5.0,
                        ),
                    ),
                )

                # Roughly 90-th quantile
                quantile = torch.mean(mean_grad_norm) + 1.6 * torch.std(mean_grad_norm)
                split_indices = torch.logical_and(mean_grad_norm > quantile, keep_mask)
                split_len = torch.sum(split_indices).item()
                split_dir = mean_grad[split_indices]

                extensions = {
                    "means": raw_means.data[split_indices] + split_dir,
                    "values": values.data[split_indices],
                    "scaling": scaling.data[split_indices],
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
                scaling = new_tensors["scaling"]

            print(split_len, torch.sum(~keep_mask).item())

            mean_grad = torch.zeros_like(raw_means, device="cuda")
            mean_grad_norm = torch.zeros(raw_means.shape[0], device="cuda")
            scale_grad = torch.zeros_like(raw_means, device="cuda")
            scale_grad_norm = torch.zeros(raw_means.shape[0], device="cuda")

        j += 1

    with torch.no_grad():
        sampler.preprocess(means, values, covariances, conics, img_samples)
        img = sampler.sample_gaussians()
        area = torch.sum(img).item() / res

        if i > 0:
            values = values * initial_area/area

    if i == 0:
        initial_area = area
        print("Area:", initial_area)

    plt.figure()
    plt.plot(np.arange(0, len(losses)*100, 100), losses)
    plt.yscale("log")
    plt.savefig("results_no_mlp_1d/loss_{}.png".format(i))

    # plt.figure()
    # plt.plot(np.arange(0, len(max_mean_grad)*100, 100), max_mean_grad)
    # plt.yscale("log")
    # plt.savefig("results_no_mlp_1d/max_mean_grad_{}.png".format(i))

    # plt.figure()
    # plt.plot(np.arange(0, len(max_mean_grad)*100, 100), max_scale_grad)
    # plt.yscale("log")
    # plt.savefig("results_no_mlp_1d/max_scale_grad_{}.png".format(i))

    means = torch.tanh(raw_means) * scale
    covariances = torch.exp(scaling)
    conics = 1.0 / covariances

    sampler.preprocess(means, values, covariances, conics, img_samples)
    results = sampler.sample_gaussians().detach().cpu().numpy()

    fig = plt.figure()
    if problem == Problem.WAVE:
        ax = fig.subplots(1, 2)
        ax[0].plot(img_samples.detach().cpu().numpy(), results[...,0])
        ax[1].plot(img_samples.detach().cpu().numpy(), results[...,1])
    else:
        plt.plot(img_samples.detach().cpu().numpy(), results)
    # if (i % 5) == 0:
    #     plt.plot(img_samples.detach().cpu().numpy(), gt[::5, i//5])
    #     plt.legend(["Prediction", "Ground Truth"])

    for k in range(means.shape[0]):
        samples = img_samples - means[k]
        powers = -0.5 * conics[k].item() * samples ** 2
        results = values[k].unsqueeze(0) * torch.exp(powers).unsqueeze(-1)
        results = results.detach().cpu().numpy()
        if problem == Problem.WAVE:
            ax[0].plot(img_samples.detach().cpu().numpy(), results[...,0], "--")
            ax[1].plot(img_samples.detach().cpu().numpy(), results[...,1], "--")
        else:
            plt.plot(img_samples.detach().cpu().numpy(), results, "--")

    # plt.gca().set_aspect(5)
    plt.savefig("results_no_mlp_1d/frame_{}_{}.png".format(problem.name.lower(), i))
    plt.close(fig)

    torch.save({
        "means": raw_means,
        "values": values,
        "scaling": scaling,
    }, "results_no_mlp_1d/gaussians_{}_{}.pt".format(problem.name.lower(), i))

    prev_means = means.detach().clone()
    prev_covariances = covariances.detach().clone()
    prev_conics = conics.detach().clone()
    prev_values = values.detach().clone()
