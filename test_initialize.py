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

nx = ny = 50
d = 2

torch.manual_seed(0)

tx = torch.linspace(-1, 1, nx).cuda()
ty = torch.linspace(-1, 1, ny).cuda()
gx, gy = torch.meshgrid((tx, ty), indexing="ij")
raw_means = torch.stack((gx, gy), dim=-1).reshape(nx*ny,d)
raw_scaling = torch.ones((nx*ny, 2), device="cuda") * -5.0
transforms = torch.zeros((nx*ny, d * (d - 1) // 2), device="cuda")
values = torch.zeros((nx*ny, 1), device="cuda")

sample_mean = None
frequency = None

if len(sys.argv) > 1:
    if sys.argv[1] == "gaussian":
        sample_mean = torch.tensor([0.2, 0.0], device="cuda").reshape(1, d, 1)
        sample_mean2 = torch.tensor([-0.6, 0.0], device="cuda").reshape(1, d, 1)
    elif sys.argv[1] == "sinusoid":
        frequency = 1.5 * np.pi
    elif sys.argv[1] == "f":
        f_index = int(sys.argv[2])

        file = np.load("ns_V1e-4_N1000_T30.npy")
        f = np.transpose(file[...,f_index], (1, 2, 0))

        # f = scipy.io.loadmat("../training_data/ns_V1e-5_N1200_T20.mat")["u"][f_index]
        desired = f[:,:,18]
        desired = torch.from_numpy(desired).to(torch.float).cuda()
        res = desired.shape[0]

        values = torch.zeros((nx*ny, 2), device="cuda")
    else:
        img = imageio.imread(sys.argv[1])
        desired = np.array(img)
        desired = torch.from_numpy(desired[:,:,0]).to(torch.float).cuda() / 255.0
        res = desired.shape[0]
else:
    res = 256
    desired = torch.zeros((res,res), device="cuda")

    # Gradient
    for i in range(res):
        desired[i,:res//2] = i / res

    # Stripe pattern
    for i in range(res//2):
        desired[i,res//2:] = (i//(res//32)) % 2

    # Checker pattern
    for i in range(res//2,res):
        for j in range(res//2,res):
            desired[i,j] = (i//(res//32)+j//(res//32)) % 2

sampler = GaussianSampler(True)

raw_means = nn.Parameter(raw_means)
values = nn.Parameter(values)
raw_scaling = nn.Parameter(raw_scaling)
transforms = nn.Parameter(transforms)

optim = torch.optim.Adam([
    { "name": "means", "params": raw_means, "lr": 5e-3 },
    { "name": "values", "params": values },
    { "name": "scaling", "params": raw_scaling, "lr": 5e-2 },
    { "name": "transforms", "params": transforms, "lr": 5e-2 }
], lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)

log_step = 100
split_step = log_step * 100 + 1
densification_step = log_step * 300 + 11

losses = []
max_mean_grad = []
max_scale_grad = []

for i in range(6000):
    samples = torch.rand((1024, d), device="cuda") * 2.0 - 1.0

    if len(sys.argv) > 1:# and sys.argv[1] == "f":
        means = raw_means
    else:
        means = torch.tanh(raw_means)

    scaling = torch.exp(raw_scaling)
    full_covariances, full_conics = gaussians.build_full_covariances(scaling, transforms)
    # full_covariances = scaling.unsqueeze(-1) \
    #                  * torch.eye(d, device="cuda").unsqueeze(0).repeat(nx*ny, 1, 1)
    # full_conics = torch.inverse(full_covariances)
    covariances, conics = gaussians.flatten_covariances(full_covariances, full_conics)

    sampler.preprocess(means, values, covariances, conics, samples)
    if len(sys.argv) > 1 and sys.argv[1] == "f":
        ux = sampler.sample_gaussians_derivative().reshape(1024, d, 2)
        img = ux[:,0,1] - ux[:,1,0]
    else:
        img = sampler.sample_gaussians().reshape(1024)

    if sample_mean is not None:
        samples_ = samples.unsqueeze(-1) - sample_mean
        conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1)
        powers = -0.5 * (samples_.transpose(-1, -2) @ (conics @ samples_))
        desired = torch.exp(powers).squeeze() * 0.5
        samples_ = samples.unsqueeze(-1) - sample_mean2
        conics = torch.inverse(torch.diag(torch.tensor([0.025, 0.1], device="cuda")))
        powers = -0.5 * (samples_.transpose(-1, -2) @ (conics @ samples_))
        desired += torch.exp(powers).squeeze()
        loss = torch.mean((img - desired) ** 2)
    elif frequency is not None:
        desired = torch.cos(frequency * samples[:,0]) * torch.cos(frequency * samples[:,1])
        loss = torch.mean((img - desired) ** 2)
    elif len(sys.argv) > 1 and sys.argv[1] == "f":
        coords = ((samples + 1.0) / 2.0 * res).to(torch.long).clamp(0, res-1)
        coords = coords[:,1] * res + coords[:,0]

        loss = torch.mean((img - torch.take(desired, coords)) ** 2)
        loss += torch.mean((ux[:,0,0] + ux[:,1,1]) ** 2)
    elif len(sys.argv) > 1:
        coords = ((samples + 1.0) / 2.0 * res).to(torch.long).clamp(0, res-1)
        coords = coords[:,1] * res + coords[:,0]

        loss = torch.mean((img - torch.take(desired, coords)) ** 2)
    else:
        coords = ((samples + 1.0) / 2.0 * res).to(torch.long).clamp(0, res-1)
        coords = coords[:,1] * res + coords[:,0]
        step = samples[:,0] > 0.0
        coords = coords[step]

        loss = torch.mean((img[step] - torch.take(desired, coords)) ** 2)
        loss += torch.mean((img[~step] - (samples[~step,1] + 1.0) / 2.0) ** 2)

    loss.backward()

    if ((i+1) % log_step) == 0:
        scheduler.step()

        mean_grad = raw_means.grad
        mean_grad_norm = torch.norm(mean_grad, dim=-1)
        scale_grad = raw_scaling.grad
        scale_grad_norm = torch.norm(scale_grad, dim=-1)
        max_mean_grad.append(torch.amax(mean_grad_norm).item())
        max_scale_grad.append(torch.amax(scale_grad_norm).item())

        # gaussians.plot_gaussians(means, covariances, values)
        # plt.show()

        print("Iteration", i)
        print("   loss", loss.item())
        print("   r", loss.item()/prev_loss)
        print()
        losses.append(loss.item())

    optim.step()
    optim.zero_grad()

    if len(sys.argv) > 1:# and sys.argv[1] == "f":
        with torch.no_grad():
            oob = raw_means > 1.0
            raw_means[oob] -= 2.0 
            oob = raw_means < -1.0
            raw_means[oob] += 2.0

    if ((i+1) % densification_step) == 0:
        means.data += torch.randn_like(means).clamp(-1, 1) * 0.01
        values.data *= 0.0

    if ((i+1) % split_step) == 0:
        if len(sys.argv) > 1:# and sys.argv[1] == "f":
            means = raw_means
        else:
            means = torch.tanh(raw_means)
        scaling = torch.exp(raw_scaling)
        full_covariances, full_conics = gaussians.build_full_covariances(scaling, transforms)
        covariances, conics = gaussians.flatten_covariances(full_covariances, full_conics)
        # gaussians.plot_gaussians(means, covariances, values)
        # plt.savefig("initialize_before.png", dpi=200)

        with torch.no_grad():
            keep_mask = torch.logical_and(
                torch.norm(values, dim=-1) > 0.01,
                torch.sum(torch.exp(raw_scaling), dim=-1) < 0.2
            )

            split_indices = torch.logical_and(mean_grad_norm > 0.0005, keep_mask)
            split_len = split_indices.sum().item()

            print("Keep:", keep_mask.sum().item(), " Split:", split_len)

            # split_dir = mean_grad[split_indices]

            eigvals, eigvecs = torch.linalg.eig(full_covariances[split_indices])
            eigval_max, indices = torch.max(eigvals.real.abs(), dim=-1, keepdim=True)
            eigvec_max = torch.gather(
                eigvecs.real.transpose(-1,-2), 1, indices.unsqueeze(-1).expand(eigvals.shape[0],1,d))
            pc = eigval_max * eigvec_max.squeeze(1) * 0.2
            raw_means.data[split_indices] = raw_means.data[split_indices] - pc
            values.data[split_indices] *= 0.5

            extensions = {
                "means": raw_means.data[split_indices] + 2 * pc,
                "values": values.data[split_indices],
                "scaling": raw_scaling.data[split_indices],
                "transforms": transforms.data[split_indices]
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
            transforms = new_tensors["transforms"]

        # means = torch.tanh(raw_means)
        # scaling = torch.exp(raw_scaling)
        # covariances = gaussians.build_covariances(scaling, transforms)
        # gaussians.plot_gaussians(means, covariances, values)
        # plt.savefig("initialize_after.png", dpi=200)
        # exit()

    prev_loss = loss.item()

# keep_mask = torch.norm(values, dim=-1) > 0.01
# print(keep_mask.sum().item())
# raw_means = raw_means[keep_mask]
# values = values[keep_mask]
# raw_scaling = raw_scaling[keep_mask]
# transforms = transforms[keep_mask]

if len(sys.argv) > 1:# and sys.argv[1] == "f":
    means = raw_means
else:
    means = torch.tanh(raw_means)
scaling = torch.exp(raw_scaling)

# print(transforms.min().item(), transforms.max().item(), transforms.mean().item()) 
# if len(sys.argv) > 1 and sys.argv[1] == "f":
#     torch.save({
#         "means": means,
#         "values": values,
#         "scaling": scaling,
#         "transforms": transforms,
#     }, "initialization/V1e-3/f_{}-10-small.pt".format(f_index))

# torch.save({
#     "means": means,
#     "values": values,
#     "scaling": scaling,
#     "transforms": transforms,
# }, "initialization/double_gaussian2.pt")

full_covariances, full_conics = gaussians.build_full_covariances(scaling, transforms)
# full_covariances = scaling.unsqueeze(-1) \
#              * torch.eye(d, device="cuda").unsqueeze(0).repeat(nx*ny, 1, 1)
# full_conics = torch.inverse(full_covariances)
covariances, conics = gaussians.flatten_covariances(full_covariances, full_conics)

gaussians.plot_gaussians(means, covariances, values)
plt.savefig("initialize_gaussians.png")
# plt.savefig("../../notes/figures/sinusoid_gaussians.png")

plt.figure()
plt.plot(np.arange(0, len(losses)*100, 100), losses)
plt.yscale("log")
plt.savefig("initialize_loss.png")

# plt.figure()
# plt.plot(np.arange(0, len(losses)*100, 100), max_mean_grad)
# plt.yscale("log")
# plt.savefig("max_mean_grad.png")
# 
# plt.figure()
# plt.plot(np.arange(0, len(losses)*100, 100), max_scale_grad)
# plt.yscale("log")
# plt.savefig("max_scale_grad.png")

# res = 256
for res in [4, 16, 32, 64, 128]:
    tx = torch.linspace(-1, 1, res).cuda()
    ty = torch.linspace(-1, 1, res).cuda()
    gx, gy = torch.meshgrid((tx, ty), indexing="xy")
    samples = torch.stack((gx, gy), dim=-1).reshape(res*res, 2)

    sampler.preprocess(means, values, covariances, conics, samples)
    if len(sys.argv) > 1 and sys.argv[1] == "f":
        ux = sampler.sample_gaussians_derivative().reshape(res, res, d, 2)
        img = ux[...,0,1] - ux[...,1,0]
    else:
        img = sampler.sample_gaussians().reshape(res, res)

    if sample_mean is not None:
        samples_ = samples.unsqueeze(-1) - sample_mean
        conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1)
        powers = -0.5 * (samples_.transpose(-1, -2) @ (conics @ samples_))
        desired = torch.exp(powers).squeeze() * 0.5
        samples_ = samples.unsqueeze(-1) - sample_mean2
        conics = torch.inverse(torch.diag(torch.tensor([0.025, 0.1], device="cuda")))
        powers = -0.5 * (samples_.transpose(-1, -2) @ (conics @ samples_))
        desired += torch.exp(powers).squeeze()
        desired = desired.reshape(res, res)
    if frequency is not None:
        desired = \
            (torch.cos(frequency * samples[:,0]) * torch.cos(frequency * samples[:,1])).reshape(res, res)

    vmin = torch.min(desired)
    vmax = torch.max(desired)

    fig = plt.figure(figsize=(9, 4))
    ax = fig.subplots(1, 2)
    im = ax[0].imshow(img.detach().cpu().numpy(), vmin=vmin, vmax=vmax, cmap="plasma")
    ax[0].axis("off")
    ax[0].invert_yaxis()
    im = ax[1].imshow(desired.detach().cpu().numpy(), vmin=vmin, vmax=vmax, cmap="plasma")
    ax[1].axis("off")
    ax[1].invert_yaxis()
    cbar_ax = fig.add_axes([0.925, 0.1, 0.025, 0.8])
    fig.colorbar(im, cax=cbar_ax)
    plt.savefig("../../notes/defense/continuous/initialize_{}.png".format(res))
# plt.savefig("../../notes/figures/sinusoid.png")
