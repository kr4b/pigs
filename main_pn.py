import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn

import gaussians
import model

from model_pn import *

from diff_gaussian_sampling import GaussianSampler

train_timesteps = 10
cutoff_timesteps = 1
test_timesteps = 15

scale = 2.5

nx = ny = 60
d = 2

torch.manual_seed(1)

model = Model(
    Problem.WAVE, 
    IntegrationRule.TRAPEZOID,
    nx, ny, d, scale
)

if model.problem == Problem.NAVIER_STOKES:
    gaussian_parameters = []
    for i in range(50):
        gaussian_parameters.append(torch.load("initialization/V1e-3/f_{}.pt".format(i)))

    with h5py.File("../training_data/ns_V1e-3_N5000_T50.mat", "r") as file:
        f = torch.tensor(np.transpose(file["u"][...,:50], (3, 1, 2, 0)), device="cuda")

    # means = gaussian_parameters[0]["means"].cuda()
    # values = gaussian_parameters[0]["values"].cuda()
    # scaling = gaussian_parameters[0]["scaling"].cuda()
    # transforms = gaussian_parameters[0]["transforms"].cuda()

    # model.set_initial_params(means, values, scaling, transforms)

    # covariances = gaussians.build_covariances(scaling, transforms)
    # conics = torch.inverse(covariances)

    # img1 = gaussians.sample_gaussians_img(
    #     means, conics, values, 64, 64, 1.0
    # ).detach().cpu().numpy()

    # samples = (torch.rand((10000, d), device="cuda") * 2.0 - 1.0) * scale

    # from diff_gaussian_sampling import GaussianSampler
    # sampler = GaussianSampler(True)

    # sampler.preprocess(means, values, covariances, conics, samples)

    # u_samples = sampler.sample_gaussians().reshape(10000)

    # desired = f[0,:,:,0]

    # res = 64
    # coords = ((samples + 1.0) / 2.0 * res).to(torch.long).clamp(0, res-1)
    # coords = coords[:,1] * res + coords[:,0]

    # loss = torch.mean((u_samples - torch.take(desired, coords)) ** 2)
    # print(loss.item())

    # img2 = f[0,:,:,0].detach().cpu().numpy()

    # fig = plt.figure()
    # ax = fig.subplots(1, 2)
    # im = ax[0].imshow(img1)
    # plt.colorbar(im)
    # im = ax[1].imshow(img2)
    # plt.colorbar(im)
    # plt.show()

    # exit()

# for name, p in model.named_parameters():
#     print(name, p.numel())

print(sum(p.numel() for p in model.parameters()))

# fig = model.plot_gaussians()
# plt.show()
# plt.close(fig)
# 
# img = model.generate_images(64)
# 
# if model.problem == Problem.WAVE:
#     plt.figure()
#     plt.imshow(img[0,:,:])
#     plt.colorbar()
#     plt.show()
# else:
#     plt.figure()
#     plt.imshow(img[:,:])
#     plt.colorbar()
#     plt.show()
# 
# exit()

training_loss = []
mean_loss = []

log_step = 100
n_samples = 1024

optim = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.9955)

dt = 1.0

start = 0

if len(sys.argv) > 1:
    state = torch.load(sys.argv[1])
    model.load_state_dict(state["model"])
    # model.set_initial_params(
    #     model.initial_means.detach().data, model.initial_u.detach().data, model.initial_scaling.detach().data, model.initial_transforms.detach().data)
    optim.load_state_dict(state["optimizer"])
    start = state["epoch"]
    training_loss = state["training_loss"]

if len(sys.argv) <= 1 or "--resume" in sys.argv:
    os.makedirs("checkpoints", exist_ok=True)

    model.train()

    torch.autograd.set_detect_anomaly(True)

    n_samples = 1024
    N = 5000
    log_step = 10
    save_step = 100
    bootstrap_rate = 50
    epsilon = 1

    current_timesteps = 1
    epoch = start

    for epoch in range(start, N):
        time_samples = torch.rand(n_samples, device="cuda")
        samples = (torch.rand((n_samples, d), device="cuda") * 2.0 - 1.0) * scale

        boundaries = torch.cat((
            -torch.ones(n_samples//4, device="cuda") - torch.rand(n_samples//4, device="cuda") * 0.5,
            torch.ones(n_samples//4, device="cuda") + torch.rand(n_samples//4, device="cuda") * 0.5
        )) * scale
        # if model.problem == Problem.NAVIER_STOKES:
        #     bc_samples = \
        #         torch.zeros((n_samples + n_samples // 4, d), device="cuda")
        # else:
        #     bc_samples = torch.zeros((n_samples, d), device="cuda")
        bc_samples = torch.zeros((n_samples, d), device="cuda")

        bc_samples[n_samples // 2:n_samples,0] = \
            (torch.rand(n_samples // 2, device="cuda") * 2.0 - 1.0) * 1.5 * scale
        bc_samples[n_samples // 2:n_samples,1] = boundaries
        bc_samples[:n_samples // 2,1] = \
            (torch.rand(n_samples // 2, device="cuda") * 2.0 - 1.0) * 1.5 * scale
        bc_samples[:n_samples // 2,0] = boundaries

        # if model.problem == Problem.NAVIER_STOKES:
        #     hypersphere = torch.rand((n_samples // 4, 1), device="cuda") * 2.0 - 1.0
        #     for i in range(d - 1):
        #         r = 1.0 - (hypersphere ** 2).sum(-1).reshape(-1, 1)
        #         hypersphere = torch.cat((
        #             hypersphere,
        #             (torch.rand((n_samples // 4, 1), device="cuda") * 2.0 - 1.0) * r
        #         ), dim=-1)

        #     bc_samples[n_samples:,:] = \
        #         (hypersphere * 0.1 - torch.tensor([[0.65, 0.0]], device="cuda")) * scale

        total_loss = 0
        total_pde_loss = 0
        total_bc_loss = 0
        total_conservation_loss = 0
        total_initial_loss = 0

        if model.problem == Problem.NAVIER_STOKES:
            total_recon_loss = 0

        model.reset()
        model.sample(samples, bc_samples)

        loss_weight = 1
        loss = torch.zeros(1, device="cuda")

        all_sufficient = True

        for i in range(min(min(epoch // bootstrap_rate + 1, current_timesteps), train_timesteps)):
            pde_loss = torch.zeros(1, device="cuda")
            bc_loss = torch.zeros(1, device="cuda")
            conservation_loss = torch.zeros(1, device="cuda")
            initial_loss = torch.zeros(1, device="cuda")

            t = i * dt

            model.forward(t, dt)
            losses = model.compute_loss(t, dt, samples, time_samples, bc_samples)

            if not torch.isnan(losses[0]) and not torch.isinf(losses[0]):
                pde_loss += losses[0]
            if not torch.isnan(losses[1]) and not torch.isinf(losses[1]):
                bc_loss += losses[1]
            if not torch.isnan(losses[2]) and not torch.isinf(losses[2]):
                conservation_loss += losses[2]
            if not torch.isnan(losses[3]) and not torch.isinf(losses[3]):
                initial_loss += losses[3]

            current_loss = pde_loss + bc_loss + conservation_loss + initial_loss
            if model.problem == Problem.NAVIER_STOKES:
                desired = f[0,:,:,i]

                res = 64
                coords = ((samples + 1.0) / 2.0 * res).to(torch.long).clamp(0, res-1)
                coords = coords[:,1] * res + coords[:,0]

                recon_loss = torch.mean(
                    (model.u_samples[-1].squeeze() - torch.take(desired, coords).squeeze()) ** 2)
                current_loss += recon_loss
                total_recon_loss += recon_loss.item()

            loss = loss_weight * current_loss

            loss.backward()
            optim.step()
            optim.zero_grad()

            print(i, current_loss.item(), loss_weight)
            loss_weight *= np.exp(-epsilon * current_loss.item())

            total_loss += current_loss.item()
            total_pde_loss += pde_loss.item()
            total_bc_loss += bc_loss.item()
            total_conservation_loss += conservation_loss.item()
            total_initial_loss += initial_loss.item()

            all_sufficient &= current_loss < 1.0

            model.detach()
            model.clear()
            model.sample(samples, bc_samples)

        if all_sufficient:
            current_timesteps = min(epoch // bootstrap_rate + 1, current_timesteps) + 1

        # if loss > 0:
        #     loss.backward()
        #     optim.step()
        #     optim.zero_grad()
        # scheduler.step()

        if (epoch+1) % log_step == 0:
            training_loss.append(total_loss / (i+1) * train_timesteps)
            print("Epoch {}: Total Loss {}".format(epoch, training_loss[-1]))
            print("  BC Loss:", total_bc_loss)
            print("  PDE Loss:", total_pde_loss)
            print("  Conservation Loss:", total_conservation_loss)
            print("  Initial Loss:", total_initial_loss)
            if model.problem == Problem.NAVIER_STOKES:
                print("  Reconstruction Loss:", total_recon_loss)

        if (epoch+1) % save_step == 0:
            torch.save({
                "epoch": epoch + 2,
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "training_loss": training_loss,
            }, "results_model_pn/{}_model_{}.pt".format(model.problem.name.lower(), epoch))

            fig = plt.figure()
            plt.plot(np.linspace(0, len(training_loss)*log_step, len(training_loss)), training_loss)
            plt.yscale("log")
            plt.savefig("results_model_pn/training_loss.png")
            plt.close(fig)

    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "training_loss": training_loss,
    }, "results_model_pn/{}_model.pt".format(model.problem.name.lower()))

model.clear()
model.reset()

imgs = []
vmin = np.inf
vmax = -np.inf

with torch.no_grad():
    for i in range(test_timesteps):
        fig = model.plot_gaussians()
        plt.savefig("results_model_pn/gaussians{}.png".format(i))
        plt.close(fig)

        res = 128
        tx = torch.linspace(-1, 1, res).cuda() * scale
        ty = torch.flip(torch.linspace(-1, 1, res).cuda().unsqueeze(-1), (0,1)).squeeze() * scale
        gx, gy = torch.meshgrid((tx, ty), indexing="xy")
        samples = torch.stack((gx, gy), dim=-1).reshape(res * res, d)

        model.sampler.preprocess(model.means, model.u, model.covariances, model.conics, samples)
        imgs.append(model.sampler.sample_gaussians().reshape(res, res).detach().cpu().numpy())

        vmin = min(vmin, np.min(imgs[-1]))
        vmax = max(vmax, np.max(imgs[-1]))

        t = i * dt
        model.forward(t, dt)

    for i in range(test_timesteps):
        fig = plt.figure()

        if model.problem == Problem.WAVE:
            ax = fig.subplots(1, 2)
            im = ax[0].imshow(imgs[i][0], vmin=vmin, vmax=vmax)
            plt.colorbar(im)
            im = ax[1].imshow(imgs[i][1], vmin=vmin, vmax=vmax)
            plt.colorbar(im)

        else:
            plt.imshow(imgs[i])#, cmap="plasma")
            plt.colorbar()
            plt.savefig("results_model_pn/frame{}.png".format(i))

        plt.close(fig)
