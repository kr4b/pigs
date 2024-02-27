import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f

from torch import nn

import gaussians
import model

from model import *

from diff_gaussian_sampling import GaussianSampler

dt = 0.1
dx = 0.05

train_timesteps = 10
cutoff_timesteps = 2
test_timesteps = 15

scale = 2.5

nx = ny = 20
d = 2

kernel_size = 5

train_opacity = False

torch.manual_seed(0)

model = Model(
    Problem.DIFFUSION,
    nx, ny, d, dx, dt, kernel_size,
    IntegrationRule.TRAPEZOID,
    train_opacity,
    scale
)

training_loss = []

optim = torch.optim.Adam(model.parameters())

start = 0

if len(sys.argv) > 1:
    state = torch.load(sys.argv[1])
    model.load_state_dict(state["model"])
    optim.load_state_dict(state["optimizer"])
    start = state["epoch"]
    training_loss = state["training_loss"]

if len(sys.argv) <= 1 or "--resume" in sys.argv:
    os.makedirs("checkpoints", exist_ok=True)

    model.train()

    torch.autograd.set_detect_anomaly(True)

    n_samples = 1024
    N = 500
    log_step = 10
    save_step = 100
    bootstrap_rate = 40
    epsilon = 1

    current_timesteps = 1

    for epoch in range(start, N):
        time_samples = torch.rand(n_samples, device="cuda")
        samples = (torch.rand((n_samples, d), device="cuda") * 2.0 - 1.0) * scale

        boundaries = torch.cat((
            -torch.ones(n_samples//4, device="cuda") - torch.rand(n_samples//4, device="cuda") * 0.5,
            torch.ones(n_samples//4, device="cuda") + torch.rand(n_samples//4, device="cuda") * 0.5
        )) * scale
        if model.problem == Problem.NAVIER_STOKES:
            bc_samples = \
                torch.zeros((n_samples + n_samples // 4, d), device="cuda")
        else:
            bc_samples = torch.zeros((n_samples, d), device="cuda")

        bc_samples[n_samples // 2:n_samples,0] = \
            (torch.rand(n_samples // 2, device="cuda") * 2.0 - 1.0) * 1.5 * scale
        bc_samples[n_samples // 2:n_samples,1] = boundaries
        bc_samples[:n_samples // 2,1] = \
            (torch.rand(n_samples // 2, device="cuda") * 2.0 - 1.0) * 1.5 * scale
        bc_samples[:n_samples // 2,0] = boundaries

        if model.problem == Problem.NAVIER_STOKES:
            hypersphere = torch.rand((n_samples // 4, 1), device="cuda") * 2.0 - 1.0
            for i in range(d - 1):
                r = 1.0 - (hypersphere ** 2).sum(-1).reshape(-1, 1)
                hypersphere = torch.cat((
                    hypersphere,
                    (torch.rand((n_samples // 4, 1), device="cuda") * 2.0 - 1.0) * r
                ), dim=-1)

            bc_samples[n_samples:,:] = \
                (hypersphere * 0.1 - torch.tensor([[0.65, 0.0]], device="cuda")) * scale

        loss = torch.zeros(1, device="cuda")

        total_loss = 0
        total_pde_loss = 0
        total_bc_loss = 0
        total_conservation_loss = 0

        model.reset()
        model.sample(samples, bc_samples)

        loss_weight = 1
        loss = torch.zeros(1, device="cuda")

        all_sufficient = True

        for i in range(min(min(epoch // bootstrap_rate + 1, current_timesteps), train_timesteps)):
            t = dt * i

            pde_loss = torch.zeros(1, device="cuda")
            bc_loss = torch.zeros(1, device="cuda")
            conservation_loss = torch.zeros(1, device="cuda")

            model.forward()
            losses = model.compute_loss(t, samples, time_samples, bc_samples)

            if not torch.isnan(losses[0]) and not torch.isinf(losses[0]):
                pde_loss += losses[0]
            if not torch.isnan(losses[1]) and not torch.isinf(losses[1]):
                bc_loss += losses[1]
            if not torch.isnan(losses[2]) and not torch.isinf(losses[2]):
                conservation_loss += losses[2]

            current_loss = pde_loss + bc_loss + conservation_loss
            loss += loss_weight * current_loss
            print(i, current_loss.item(), loss_weight)
            loss_weight *= np.exp(-epsilon * current_loss.item())

            total_loss += current_loss.item()
            total_pde_loss += pde_loss.item()
            total_bc_loss += bc_loss.item()
            total_conservation_loss += conservation_loss.item()

            all_sufficient &= current_loss < 1.0

            if (i+1) % cutoff_timesteps == 0:
                loss.backward()
                optim.step()
                optim.zero_grad()

                loss = torch.zeros(1, device="cuda")

                model.detach()
                model.clear()
                model.sample(samples, bc_samples)

        if all_sufficient:
            current_timesteps = min(epoch // bootstrap_rate + 1, current_timesteps) + 1

        if loss > 0:
            loss.backward()
            optim.step()
            optim.zero_grad()

        if (epoch+1) % log_step == 0:
            training_loss.append(total_loss / (i+1) * train_timesteps)
            print("Epoch {}: Total Loss {}".format(epoch, training_loss[-1]))
            print("  BC Loss:", total_bc_loss)
            print("  PDE Loss:", total_pde_loss)
            print("  Conservation Loss:", total_conservation_loss)

        if (epoch+1) % save_step == 0:
            torch.save({
                "epoch": epoch + 2,
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "training_loss": training_loss,
            }, "checkpoints/{}_model_{}.pt".format(model.problem.name.lower(), epoch))

            fig = plt.figure()
            plt.plot(np.linspace(0, len(training_loss)*log_step, len(training_loss)), training_loss)
            plt.yscale("log")
            plt.savefig("training_loss.png")
            plt.close(fig)

    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "training_loss": training_loss,
    }, "{}_model.pt".format(model.problem.name.lower()))

res = 128
os.makedirs("results", exist_ok=True)

model.reset()
model.eval()

with torch.no_grad():
    imgs1 = []
    imgs2 = []
    imgs3 = []
    imgs4 = []
    uxxs = []
    us = []

    vmax = -np.inf
    zmax = -np.inf
    vmin = np.inf
    zmin = np.inf

    sampler = GaussianSampler(False)

    for i in range(test_timesteps + 1):
        t = i * dt

        fig = model.plot_gaussians()
        plt.savefig("results/gaussians_plot{}.png".format(i))
        plt.close(fig)

        tx = torch.linspace(-1, 1, res).cuda() * scale * 2.0
        ty = torch.linspace(-1, 1, res).cuda() * scale * 2.0
        gx, gy = torch.meshgrid((tx,ty), indexing="ij")

        samples = torch.stack((gx, gy), dim=-1).reshape(res*res, d)

        img1, img2, img3, img4 = model.generate_images(res)

        imgs1.append(img1)
        imgs2.append(img2)
        imgs3.append(img3)
        imgs4.append(img4)

        sampler.preprocess(
            model.means, model.u, model.covariances, model.conics,
            torch.sigmoid(model.opacities), samples)

        uxx = sampler.sample_gaussians_laplacian() # n, d, d, c
        uxxs.append(uxx.reshape(res, res, d, d, -1).detach().cpu().numpy())

        if i < test_timesteps:
            model.forward()

    for i in range(train_timesteps + 1):
        if model.problem == Problem.WAVE:
            vmin = min(vmin, np.min(imgs2[i]), np.min(imgs4[i]))
            zmin = min(zmin, np.min(imgs1[i]), np.min(imgs3[i]))
            vmax = max(vmax, np.max(imgs2[i]), np.max(imgs4[i]))
            zmax = max(zmax, np.max(imgs1[i]), np.max(imgs3[i]))
        else:
            vmin = min(vmin, np.min(imgs1[i]), np.min(imgs2[i]), np.min(imgs3[i]), np.min(imgs4[i]))
            vmax = max(vmax, np.max(imgs1[i]), np.max(imgs2[i]), np.max(imgs3[i]), np.max(imgs4[i]))

    for i in range(test_timesteps + 1):

        if model.problem == Problem.NAVIER_STOKES:
            fig = plt.figure(figsize=(6,6))
            ax = fig.subplots(3, 2)
        else:
            fig = plt.figure()
            ax = fig.subplots(2, 2)

        if model.problem == Problem.WAVE:
            ax[0,0].set_title("Initial z")
            ax[0,1].set_title("Initial v")
            ax[1,0].set_title("z")
            ax[1,1].set_title("v")
        elif model.problem == Problem.NAVIER_STOKES:
            ax[0,0].set_title("Initial v")
            ax[0,1].set_title("Initial p")
            ax[1,0].set_title("v")
            ax[1,1].set_title("p")
            ax[2,0].set_title("v_x")
            ax[2,1].set_title("v_y")
        else:
            ax[0,0].set_title("Initial")
            ax[0,1].set_title("Full")
            ax[1,0].set_title("Only u")
            ax[1,1].set_title("Only G")

        if model.problem == Problem.WAVE:
            im = ax[0,0].imshow(imgs1[i], vmin=zmin, vmax=zmax)
            im = ax[0,1].imshow(imgs2[i], vmin=vmin, vmax=vmax)
            im = ax[1,0].imshow(imgs3[i], vmin=zmin, vmax=zmax)
            im = ax[1,1].imshow(imgs4[i], vmin=vmin, vmax=vmax)
        elif model.problem == Problem.NAVIER_STOKES:
            im = ax[0,0].imshow(imgs1[i], vmin=vmin, vmax=vmax)
            im = ax[0,1].imshow(imgs2[i], vmin=vmin, vmax=vmax)
            im = ax[1,0].imshow(imgs3[i][...,2], vmin=vmin, vmax=vmax)
            im = ax[1,1].imshow(imgs4[i], vmin=vmin, vmax=vmax)
            im = ax[2,0].imshow(imgs3[i][...,0], vmin=vmin, vmax=vmax)
            im = ax[2,1].imshow(imgs3[i][...,1], vmin=vmin, vmax=vmax)
        else:
            im = ax[0,0].imshow(imgs1[i], vmin=vmin, vmax=vmax)
            im = ax[0,1].imshow(imgs2[i], vmin=vmin, vmax=vmax)
            im = ax[1,0].imshow(imgs3[i], vmin=vmin, vmax=vmax)
            im = ax[1,1].imshow(imgs4[i], vmin=vmin, vmax=vmax)

        cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
        fig.colorbar(im, cax=cbar_ax)
        plt.tight_layout()
        plt.savefig("results/results{}.png".format(i), dpi=200)
        plt.close(fig)

        if model.problem == Problem.NAVIER_STOKES or model.problem == Problem.WAVE:
            fig = plt.figure()
            ax = fig.subplots(2, 2)
            im = ax[0,0].imshow(uxxs[i][...,0,0,0])
            plt.colorbar(im)
            im = ax[0,1].imshow(uxxs[i][...,1,1,0])
            plt.colorbar(im)
            im = ax[1,0].imshow(uxxs[i][...,0,0,1])
            plt.colorbar(im)
            im = ax[1,1].imshow(uxxs[i][...,1,1,1])
            plt.colorbar(im)
            plt.savefig("results/uxx{}.png".format(i), dpi=200)
            plt.close(fig)
