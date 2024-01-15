import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as f

from torch import nn

import gaussians
import model

from model import *

nu = 1.0 / (100.0 * np.pi)
dt = 0.01
dx = 0.1

train_timesteps = 10
test_timesteps = 20

nx = ny = 20
d = 2

kernel_size = 5

model = Model(Problem.DIFFUSION, nx, ny, d, dx, dt, kernel_size, IntegrationRule.TRAPEZOID)

if len(sys.argv) > 1:
    model.load_state_dict(torch.load(sys.argv[1]))

if len(sys.argv) <= 1 or "--resume" in sys.argv:
    os.makedirs("checkpoints", exist_ok=True)

    optim = torch.optim.Adam(model.parameters())
    model.train()

    torch.autograd.set_detect_anomaly(True)

    N = 1000
    log_step = N / 100
    save_step = N / 10
    bootstrap_rate = 4
    start = 0
    training_error = []

    loss_cutoff = 0

    torch.manual_seed(0)

    for epoch in range(N):
        samples = torch.rand((100, 2), device="cuda") * 2.0 - 1.0

        boundaries = torch.cat((-torch.ones(25, device="cuda"), torch.ones(25, device="cuda")))
        bc_samples = torch.zeros((100, 2), device="cuda")
        bc_samples[50:,0] = torch.rand(50, device="cuda") * 2.0 - 1.0
        bc_samples[50:,1] = boundaries
        bc_samples[:50,1] = torch.rand(50, device="cuda") * 2.0 - 1.0
        bc_samples[:50,0] = boundaries

        pde_loss = torch.zeros(1, device="cuda")
        bc_loss = torch.zeros(1, device="cuda")
        conservation_loss = torch.zeros(1, device="cuda")

        model.reset()
        model.sample(samples, bc_samples)

        for i in range(min(epoch // bootstrap_rate + 1, train_timesteps)):
            t = dt * i

            model.forward(dt)
            losses = model.compute_loss(t, samples, bc_samples)

            pde_loss += losses[0]
            bc_loss += losses[1]
            conservation_loss += losses[2]

        loss = pde_loss + bc_loss + conservation_loss

        loss.backward()
        optim.step()
        optim.zero_grad()

        if loss.item() < loss_cutoff:
            break

        if epoch % log_step == 0:
            training_error.append(loss.item())
            print("Epoch {}: Total Loss {}".format(epoch, training_error[-1]))
            print("  BC Loss:", bc_loss.item())
            print("  PDE Loss:", pde_loss.item())
            print("  Conservation Loss:", conservation_loss.item())
            print("  Deltas:", torch.median(model.deltas).item(), torch.mean(model.deltas).item(), torch.min(model.deltas).item(), torch.max(model.deltas).item())

        if epoch % save_step == 0:
            torch.save(model.state_dict(), "checkpoints/{}_model_{}.pt".format(model.problem.name.lower(), start + epoch))

    plt.figure()
    plt.plot(np.linspace(0, len(training_error) * log_step, len(training_error)), training_error)
    plt.yscale("log")
    plt.savefig("training_error.png")

    torch.save(model.state_dict(), "{}_model.pt".format(problem.name.lower()))

res = 32
os.makedirs("results", exist_ok=True)

model.reset()
model.eval()

with torch.no_grad():
    imgs1 = []
    imgs2 = []
    imgs3 = []
    imgs4 = []

    vmax = -np.inf
    vmin = np.inf

    for i in range(test_timesteps):
        t = i * dt

        fig = model.plot_gaussians()
        plt.savefig("results/gaussians_plot{}.png".format(i))
        plt.close(fig)

        img1, img2, img3, img4 = model.generate_images(res)
        model.forward(dt)

        imgs1.append(img1)
        imgs2.append(img2)
        imgs3.append(img3)
        imgs4.append(img4)

    for i in range(train_timesteps):
        vmax = max(vmax, np.max(imgs1[i]), np.max(imgs2[i]), np.max(imgs3[i]), np.max(imgs4[i]))
        vmin = min(vmin, np.min(imgs1[i]), np.min(imgs2[i]), np.min(imgs3[i]), np.min(imgs4[i]))

    for i in range(test_timesteps):
        fig = plt.figure()
        ax = fig.subplots(2, 2)
        im = ax[0,0].imshow(imgs1[i], vmin=vmin, vmax=vmax)
        im = ax[0,1].imshow(imgs2[i], vmin=vmin, vmax=vmax)
        im = ax[1,0].imshow(imgs3[i], vmin=vmin, vmax=vmax)
        im = ax[1,1].imshow(imgs4[i], vmin=vmin, vmax=vmax)
        cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig("results/results{}.png".format(i))
        plt.close(fig)
