import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import imageio.v3 as imageio

from torch import nn

import gaussians

from diff_gaussian_sampling import GaussianSampler

os.makedirs("results_initialize_1d", exist_ok=True)

n = 10
d = 1

torch.manual_seed(0)

raw_means = torch.linspace(-1, 1, n).cuda().reshape(-1, d)
scaling = torch.ones((n,d), device="cuda") * -5.0

values = torch.rand((n, d), device="cuda")

raw_means = nn.Parameter(raw_means)
values = nn.Parameter(values)
scaling = nn.Parameter(scaling)

sampler = GaussianSampler(True)
optim = torch.optim.Adam([
    { "name": "means", "params": raw_means },
    { "name": "values", "params": values },
    { "name": "scaling", "params": scaling },
], lr=1e-2)

log_step = 100
densification_step = log_step * 10 + 1

losses = []
max_mean_grad = []
max_scale_grad = []

sample_mean = 0.5

res = 200
img_samples = torch.linspace(-1, 1, res).cuda()

for i in range(10000):
    samples = torch.rand((128, d), device="cuda") * 2.0 - 1.0

    means = torch.tanh(raw_means)
    covariances = torch.exp(scaling).reshape(-1, d, d)
    conics = 1.0 / covariances

    sampler.preprocess(means, values, covariances, conics, samples)
    img = sampler.sample_gaussians()

    powers = -5.0 * (samples - sample_mean) ** 2
    desired = torch.exp(powers)
    loss = torch.mean((img - desired) ** 2)

    loss.backward()

    if ((i+1) % log_step) == 0:
        mean_grad = raw_means.grad
        mean_grad_norm = torch.norm(mean_grad, dim=-1)
        scale_grad = scaling.grad
        scale_grad_norm = torch.norm(scale_grad, dim=-1)
        max_mean_grad.append(torch.amax(mean_grad_norm).item())
        max_scale_grad.append(torch.amax(scale_grad_norm).item())

        print("Iteration", i)
        print("   loss", loss.item())
        print("   r", loss.item()/prev_loss)
        print()
        losses.append(loss.item())

        sampler.preprocess(means, values, covariances, conics, img_samples)
        results = sampler.sample_gaussians().detach().cpu().numpy()

        fig = plt.figure()
        plt.plot(img_samples.detach().cpu().numpy(), results)
        for j in range(means.shape[0]):
            samples = img_samples - means[j]
            powers = -0.5 * conics[j].item() * samples ** 2
            results = values[j] * torch.exp(powers)
            results = results.detach().cpu().numpy()
            plt.plot(img_samples.detach().cpu().numpy(), results, "--")

        plt.savefig("results_initialize_1d/frame_{}.png".format(i//log_step))
        plt.close(fig)

    optim.step()
    optim.zero_grad()

    if ((i+1) % densification_step) == 0:
        with torch.no_grad():
            keep_mask = torch.logical_and(
                torch.norm(values, dim=-1) > 0.01,
                torch.sum(torch.exp(scaling), dim=-1) < 0.1
            )

            split_indices = torch.logical_and(mean_grad_norm > 0.01, keep_mask)
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
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"][keep_mask], torch.zeros_like(extension)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"][keep_mask], torch.zeros_like(extension)), dim=0)

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

    prev_loss = loss.item()
