import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f

import imageio.v3 as imageio

from torch import nn

import gaussians

from diff_gaussian_sampling import GaussianSampler

res = 32
nx = ny = 3
d = 2

tx = torch.linspace(-1, 1, nx).cuda()
ty = torch.linspace(-1, 1, ny).cuda()
gx, gy = torch.meshgrid((tx,ty), indexing="ij")
raw_means = torch.stack((gx,gy), dim=-1).reshape(nx*ny,d)
raw_scaling = torch.ones((nx*ny,d), device="cuda") * -4.0
raw_transform = torch.zeros((nx*ny,d * (d - 1) // 2), device="cuda")
opacities = torch.ones((nx*ny), device="cuda")
values = torch.ones((nx*ny,1), device="cuda")

tx = torch.linspace(-1, 1, res).cuda()
ty = torch.linspace(-1, 1, res).cuda()
gx, gy = torch.meshgrid((tx, ty), indexing="xy")
samples = torch.stack((gx, gy), dim=-1).reshape(res * res, 2)

raw_means = nn.Parameter(raw_means)
values = nn.Parameter(values)
raw_scaling = nn.Parameter(raw_scaling)
raw_transform = nn.Parameter(raw_transform)

sample_mean = None

if len(sys.argv) > 1:
    if sys.argv[1] == "gaussian":
        sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, d, 1)
    else:
        img = imageio.imread(sys.argv[1])
        desired = np.array(img)
        desired = torch.from_numpy(desired[:,:,0]).to(torch.float).cuda() / 255.0
        res = desired.shape[0]
else:
    desired = torch.zeros((res,res), device="cuda")

    # Gradient
    for i in range(res):
        desired[i,:res//2] = i / res

    # Stripe pattern
    for i in range(res//2):
        desired[i,res//2:] = i % 2

    # Checker pattern
    for i in range(res//2,res):
        for j in range(res//2,res):
            desired[i,j] = (i+j) % 2

sampler = GaussianSampler(True)
optim = torch.optim.Adam([
    { "name": "means", "params": raw_means },
    { "name": "values", "params": values },
    { "name": "scaling", "params": raw_scaling },
    { "name": "transform", "params": raw_transform }
])

log_step = 100
densification_step = log_step * 20

losses = []
max_mean_grad = []
max_scale_grad = []

for i in range(99999):
    samples = torch.rand((1024, 2), device="cuda") * 2.0 - 1.0

    means = f.tanh(raw_means)
    scaling = torch.exp(raw_scaling)
    transform = f.tanh(raw_transform)
    covariances = gaussians.build_covariances(scaling, transform)
    conics = torch.inverse(covariances)

    sampler.preprocess(means, values, covariances, conics, opacities, samples)
    img = sampler.sample_gaussians().reshape(1024)
    # img = sampler.sample_gaussians().reshape(res, res)

    if sample_mean is not None:
        samples = samples.unsqueeze(-1) - sample_mean
        conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1)
        powers = -0.5 * (samples.transpose(-1, -2) @ (conics @ samples))
        desired = torch.exp(powers).squeeze()
        loss = torch.mean((img - desired) ** 2)
    else:
        coords = ((samples + 1.0) / 2.0 * res).to(torch.long).clamp(0, res-1)
        coords = coords[:,1] * res + coords[:,0]

        loss = torch.mean((img - torch.take(desired, coords)) ** 2)
        # loss = torch.mean((img - desired) ** 2)

    loss.backward()

    if ((i+1) % log_step) == 0:
        mean_grad = raw_means.grad
        mean_grad_norm = torch.norm(mean_grad, dim=-1)
        scale_grad = raw_scaling.grad
        scale_grad_norm = torch.norm(scale_grad, dim=-1)
        max_mean_grad.append(torch.amax(mean_grad_norm).cpu().numpy())
        max_scale_grad.append(torch.amax(scale_grad_norm).cpu().numpy())

        print("Iteration", i)
        print("   loss", loss.item())
        print("   r", loss.item()/prev_loss)
        print()
        losses.append(loss.item())

    optim.step()
    optim.zero_grad()

    if ((i+1) % densification_step) == 0:
        with torch.no_grad():
            keep_mask = torch.logical_and(
                torch.norm(values, dim=-1) > 0.01,
                torch.sum(torch.exp(raw_scaling), dim=-1) < 0.1
            )

            split_indices = torch.logical_and(mean_grad_norm > 0.01, keep_mask)
            split_len = torch.sum(split_indices).item()
            split_dir = mean_grad[split_indices]

            extensions = {
                "means": raw_means.data[split_indices] + split_dir,
                "values": values.data[split_indices],
                "scaling": raw_scaling.data[split_indices],
                "transform": raw_transform.data[split_indices]
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
            raw_scaling = new_tensors["scaling"]
            raw_transform = new_tensors["transform"]
            opacities = torch.cat((opacities, torch.ones(split_len, device="cuda")))

    prev_loss = loss.item()

means = f.tanh(raw_means)
scaling = torch.exp(raw_scaling)
transform = f.tanh(raw_transform)
covariances = gaussians.build_covariances(scaling, transform)
conics = torch.inverse(covariances)

gaussians.plot_gaussians(means, covariances, opacities * 0.25, values)
plt.savefig("initialize_gaussians.png", dpi=200)

plt.figure()
plt.plot(np.arange(0, len(losses)*100, 100), losses)
plt.yscale("log")
plt.savefig("initialize_loss.png")

plt.figure()
plt.plot(np.arange(0, len(losses)*100, 100), max_mean_grad)
plt.yscale("log")
plt.savefig("max_mean_grad.png")

plt.figure()
plt.plot(np.arange(0, len(losses)*100, 100), max_scale_grad)
plt.yscale("log")
plt.savefig("max_scale_grad.png")

res = 256

tx = torch.linspace(-1, 1, res).cuda()
ty = torch.linspace(-1, 1, res).cuda()
gx, gy = torch.meshgrid((tx, ty), indexing="xy")
samples = torch.stack((gx, gy), dim=-1).reshape(res*res, 2)

sampler.preprocess(means, values, covariances, conics, opacities, samples)
img = sampler.sample_gaussians().reshape(res,res)

if sample_mean is not None:
    samples = samples.unsqueeze(-1) - sample_mean
    conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1)
    powers = -0.5 * (samples.transpose(-1, -2) @ (conics @ samples))
    desired = torch.exp(powers).reshape(res,res)

vmin = torch.min(desired)
vmax = torch.max(desired)

fig = plt.figure(figsize=(9,4))
ax = fig.subplots(1, 2)
im = ax[0].imshow(img.detach().cpu().numpy(), vmin=vmin, vmax=vmax)
ax[0].axis("off")
ax[0].invert_yaxis()
im = ax[1].imshow(desired.detach().cpu().numpy(), vmin=vmin, vmax=vmax)
ax[1].axis("off")
ax[1].invert_yaxis()
cbar_ax = fig.add_axes([0.925, 0.1, 0.025, 0.8])
fig.colorbar(im, cax=cbar_ax)
plt.savefig("initialize.png")
