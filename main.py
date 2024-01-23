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

dt = 0.1
dx = 0.05

train_timesteps = 10
cutoff_timesteps = 2
test_timesteps = 20

n_samples = 100

nx = ny = 20
d = 2

kernel_size = 5

train_opacity = True

model = Model(
    Problem.WAVE,
    nx, ny, d, dx, dt, kernel_size,
    IntegrationRule.TRAPEZOID,
    train_opacity
)

optim = torch.optim.Adam(model.parameters())

start = 0

if len(sys.argv) > 1:
    state = torch.load(sys.argv[1])
    model.load_state_dict(state["model"])
    optim.load_state_dict(state["optimizer"])
    start = state["epoch"]

if len(sys.argv) <= 1 or "--resume" in sys.argv:
    os.makedirs("checkpoints", exist_ok=True)

    model.train()

    torch.autograd.set_detect_anomaly(True)

    N = 1000
    log_step = 10
    save_step = 100
    bootstrap_rate = 20
    epsilon = 1
    training_error = []

    torch.manual_seed(0)

    for epoch in range(start, N):
        time_samples = torch.rand(n_samples, device="cuda")
        samples = torch.rand((n_samples, 2), device="cuda") * 2.0 - 1.0

        boundaries = torch.cat((-torch.ones(n_samples // 4, device="cuda"), torch.ones(n_samples // 4, device="cuda")))
        bc_samples = torch.zeros((n_samples, 2), device="cuda")
        bc_samples[n_samples // 2:,0] = torch.rand(n_samples // 2, device="cuda") * 2.0 - 1.0
        bc_samples[n_samples // 2:,1] = boundaries
        bc_samples[:n_samples // 2,1] = torch.rand(n_samples // 2, device="cuda") * 2.0 - 1.0
        bc_samples[:n_samples // 2,0] = boundaries

        loss = torch.zeros(1, device="cuda")
        pde_loss = torch.zeros(1, device="cuda")
        bc_loss = torch.zeros(1, device="cuda")
        conservation_loss = torch.zeros(1, device="cuda")

        total_loss = 0
        total_pde_loss = 0
        total_bc_loss = 0
        total_conservation_loss = 0

        model.reset()
        model.sample(samples, bc_samples)

        loss_weight = 1

        for i in range(min(epoch // bootstrap_rate + 1, train_timesteps)):
            t = dt * i

            model.forward()
            losses = model.compute_loss(t, samples, time_samples, bc_samples)

            for j in range(len(losses)):
                if torch.isnan(losses[j]) or torch.isinf(losses[j]):
                    losses[j] = torch.zeros(1, device="cuda")

            pde_loss += losses[0]
            bc_loss += losses[1]
            conservation_loss += losses[2]

            current_loss = pde_loss + bc_loss + conservation_loss
            loss += loss_weight * current_loss
            print(i, current_loss.item(), loss_weight)
            loss_weight *= np.exp(-epsilon * current_loss.item())

            total_loss += current_loss.item()
            total_pde_loss += pde_loss.item()
            total_bc_loss += bc_loss.item()
            total_conservation_loss += conservation_loss.item()

            pde_loss = torch.zeros(1, device="cuda")
            bc_loss = torch.zeros(1, device="cuda")
            conservation_loss = torch.zeros(1, device="cuda")

            if (i+1) % cutoff_timesteps == 0:
                model.detach()
                model.clear()
                model.sample(samples, bc_samples)

        loss.backward()
        optim.step()
        optim.zero_grad()

        if (epoch+1) % log_step == 0:
            training_error.append(total_loss)
            print("Epoch {}: Total Loss {}".format(epoch, training_error[-1]))
            print("  BC Loss:", total_bc_loss)
            print("  PDE Loss:", total_pde_loss)
            print("  Conservation Loss:", total_conservation_loss)

        if (epoch+1) % save_step == 0:
            torch.save({
                "epoch": epoch + 2,
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
            }, "checkpoints/{}_model_{}.pt".format(model.problem.name.lower(), epoch))

            plt.figure()
            plt.plot(np.linspace(0, len(training_error) * log_step, len(training_error)), training_error)
            plt.yscale("log")
            plt.savefig("training_error.png")

        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
        }, "{}_model.pt".format(model.problem.name.lower()))

res = 128
os.makedirs("results", exist_ok=True)

model.reset()
model.eval()

if True:
# with torch.no_grad():
    imgs1 = []
    imgs2 = []
    imgs3 = []
    imgs4 = []
    uxxs = []
    us = []

    vmax = -np.inf
    vmin = np.inf

    for i in range(test_timesteps):
        t = i * dt

        # tx = torch.linspace(-2, 2, res).cuda()
        # ty = torch.linspace(-2, 2, res).cuda()
        # gx, gy = torch.meshgrid((tx, ty), indexing="xy")
        # img_samples = torch.stack((gx, gy), dim=-1).reshape(res * res, 2)
        # img_samples.requires_grad = True

        # u = gaussians.sample_gaussians(
        #     model.means, model.inv_sqrt_det, model.conics, model.opacities, model.u, img_samples
        # )#.sum(dim=(1,2)).reshape(res, res, 2).detach().cpu().numpy()

        # us.append(u)

        # uxx = gaussians.gaussian_derivative2(
        #     model.means, model.inv_sqrt_det, model.conics, model.opacities, model.u, img_samples
        # ).sum(dim=(1,2)).reshape(res, res, 2, 2, 2).detach().cpu().numpy()

        # grad1 = torch.autograd.grad(u[...,0].sum(), img_samples, retain_graph=True, create_graph=True)[0]
        # grad2_1 = torch.autograd.grad(grad1[:,0].sum(), img_samples, retain_graph=True)[0]
        # grad2_2 = torch.autograd.grad(grad1[:,1].sum(), img_samples, retain_graph=True)[0]
        # uxx0 = torch.cat((grad2_1, grad2_2), dim=-1)
        # grad1 = torch.autograd.grad(u[...,1].sum(), img_samples, retain_graph=True, create_graph=True)[0]
        # grad2_1 = torch.autograd.grad(grad1[:,0].sum(), img_samples, retain_graph=True)[0]
        # grad2_2 = torch.autograd.grad(grad1[:,1].sum(), img_samples)[0]
        # uxx1 = torch.cat((grad2_1, grad2_2), dim=-1)

        # uxx = torch.cat((uxx0, uxx1), dim=-1).reshape(res, res, 2, 2, 2).transpose(-3, -1).detach().cpu().numpy()

        # uxxs.append(uxx)

        # fig = model.plot_gaussians()
        # plt.savefig("results/gaussians_plot{}.png".format(i))
        # plt.close(fig)

        with torch.no_grad():
            img1, img2, img3, img4 = model.generate_images(res)
            model.forward()

        imgs1.append(img1)
        imgs2.append(img2)
        imgs3.append(img3)
        imgs4.append(img4)

    for i in range(test_timesteps):
        vmax = max(vmax, np.max(imgs1[i]), np.max(imgs2[i]), np.max(imgs3[i]), np.max(imgs4[i]))
        vmin = min(vmin, np.min(imgs1[i]), np.min(imgs2[i]), np.min(imgs3[i]), np.min(imgs4[i]))

    for i in range(test_timesteps):
        fig = plt.figure()
        ax = fig.subplots(2, 2)
        if model.problem == Problem.WAVE:
            ax[0,0].set_title("Initial z")
            ax[0,1].set_title("Initial v")
            ax[1,0].set_title("z")
            ax[1,1].set_title("v")
        else:
            ax[0,0].set_title("Initial")
            ax[0,1].set_title("Full")
            ax[1,0].set_title("Only u")
            ax[1,1].set_title("Only G")
        # im = ax[0,0].imshow(uxxs[i][...,0,0,0] + uxxs[i][...,1,1,0], vmin=vmin, vmax=vmax)
        # im = ax[0,1].imshow(uxxs[i][...,0,0,1] + uxxs[i][...,1,1,1], vmin=vmin, vmax=vmax)
        im = ax[0,0].imshow(imgs1[i], vmin=vmin, vmax=vmax)
        im = ax[0,1].imshow(imgs2[i], vmin=vmin, vmax=vmax)
        if i > 0:
            im = ax[1,0].imshow(imgs3[i], vmin=vmin, vmax=vmax)
        else:
            im = ax[1,0].imshow(imgs3[i], vmin=vmin, vmax=vmax)
        im = ax[1,1].imshow(imgs4[i], vmin=vmin, vmax=vmax)
        cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
        fig.colorbar(im, cax=cbar_ax)
        plt.tight_layout()
        plt.savefig("results/results{}.png".format(i))
        plt.close(fig)
