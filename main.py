import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as f

from torch import nn

import gaussians

problem = "DIFFUSION"
# problem = "BURGERS"
# problem = "BURGERS"
# problem = "WAVE"

class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__() 
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

class Network(nn.Module):
    def __init__(self, out_channels, activation):
        super().__init__()
        # self.dx = nn.Parameter(torch.ones(1, device="cuda") * 0.1)

        self.samples_conv = nn.Sequential(
            nn.Conv2d(channels, 64, 5),
            # nn.LayerNorm((64, 1, 1)),
            activation,
            nn.Flatten(),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(2, 32),
            # nn.LayerNorm(32),
            activation,
        )
        self.linear = nn.Sequential(
            nn.Linear(96, 128),
            # nn.LayerNorm(64),
            activation,
            nn.Linear(128, 128),
            # nn.LayerNorm(32),
            activation,
            nn.Linear(128, out_channels),
        )

    def forward(self, img, x):
        y = self.samples_conv(img)
        embed = self.pos_embed(x)
        y = self.linear(torch.cat((y, embed), dim=1))
        return y

nu = 1.0 / (100.0 * np.pi)
dt = 0.01
dx = 0.1

train_timesteps = 10
test_timesteps = 20

nx = ny = 20
d = 2

channels = 2 if problem == "WAVE" else 1

tx = torch.linspace(-1, 1, nx).cuda()
ty = torch.linspace(-1, 1, ny).cuda()
gx, gy = torch.meshgrid((tx,ty), indexing="xy")
means = torch.stack((gx,gy), dim=-1)
scaling = torch.ones((nx,ny,d), device="cuda") * -4.0
transform = torch.zeros((nx,ny,d * (d - 1) // 2), device="cuda")
opacities = torch.ones((nx,ny), device="cuda") * 0.5
conic = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1)

inv_sqrt_pi = np.power(1.0 / np.sqrt(2.0 * np.pi), d)

if problem == "POISSON":
    u = torch.zeros((nx, ny), device="cuda")
    # u = torch.sin(np.pi * (means[...,0] + 1.0))
    # u = torch.sin(np.pi * (means[...,0] + 1.0) * (means[...,1] + 1.0))
elif problem == "WAVE":
    u = torch.zeros((nx, ny), device="cuda")
    u[nx//2-1,ny//2-1] = 1.0
    u[nx//2-1,ny//2] = 1.0
    u[nx//2,ny//2-1] = 1.0
    u[nx//2,ny//2] = 1.0
else:
    sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, 1, d, 1)
    samples = means.unsqueeze(-1) - sample_mean
    powers = -0.5 * (samples.transpose(-1, -2) @ (conic @ samples))
    u = torch.exp(powers).squeeze()
    u = u / torch.max(u)

u = u.unsqueeze(-1).repeat(1, 1, channels)

scaling = torch.exp(scaling)
transform = f.tanh(transform)

translation_model = Network(d, nn.SiLU()).cuda()
scale_model = Network(d, nn.SiLU()).cuda()
transform_model = Network(transform.shape[-1], nn.SiLU()).cuda()

models = nn.ModuleList([
    translation_model,
    scale_model,
    transform_model,
])

if problem == "WAVE":
    model1 = Network(1, WaveAct()).cuda()
    model2 = Network(1, WaveAct()).cuda()
    models.append(model1)
    models.append(model2)
else:
    model = Network(1, WaveAct()).cuda()
    models.append(model)

if len(sys.argv) > 1:
    models.load_state_dict(torch.load(sys.argv[1]))

if len(sys.argv) <= 1 or "--resume" in sys.argv:
    os.makedirs("checkpoints", exist_ok=True)

    optim = torch.optim.Adam(models.parameters())
    models.train()

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

        new_means = means.clone()
        new_scaling = scaling.clone()
        new_transform = transform.clone()
        new_u = u.clone()

        pde_loss = torch.zeros(1, device="cuda")
        bc_loss = torch.zeros(1, device="cuda")
        # conservation_loss = torch.zeros(1, device="cuda")

        for i in range(min(epoch // bootstrap_rate + 1, train_timesteps)):
            t = dt * i

            covariances = gaussians.build_covariances(new_scaling, new_transform)
            conics = torch.inverse(covariances)
            inv_sqrt_det = inv_sqrt_pi * torch.sqrt(torch.det(conics))

            in_samples = gaussians.sample_gaussians_region(
                    new_means, inv_sqrt_det, conics, opacities, new_u, new_means, (5, 5), (dx, dx)) # nx*ny*5*5, nx, ny, c
            in_samples = in_samples.reshape(nx*ny * 5 * 5, nx*ny, channels).transpose(0, 1).reshape(nx*ny, channels, 5, 5, nx, ny).sum((-1, -2)) # nx*ny, c, 5, 5

            if problem == "WAVE":
                delta1 = model1(in_samples, new_means.reshape(nx*ny, d)).reshape(nx, ny)
                delta2 = model2(in_samples, new_means.reshape(nx*ny, d)).reshape(nx, ny)
                deltas = torch.stack((delta1, delta2), dim=-1) / dt
                next_u = torch.stack((new_u[...,0] + delta1, new_u[...,1] + delta2), dim=-1)
            else:
                delta = model(in_samples, new_means.reshape(nx*ny, d)).reshape(nx, ny, 1)
                deltas = delta / dt
                next_u = new_u + delta

            translation = translation_model(in_samples, new_means.reshape(nx*ny, d)).reshape(nx, ny, d)
            next_means = new_means + translation

            scale = scale_model(in_samples, new_means.reshape(nx*ny, d)).reshape(nx,ny,d)
            scale = torch.exp(scale)
            next_scaling = new_scaling * scale

            t = transform_model(in_samples, new_means.reshape(nx*ny, d)).reshape(nx,ny,-1)
            next_transform = new_transform + t

            # angle = np.pi * f.tanh(angle)
            # s = torch.sin(angle)
            # c = torch.cos(angle)

            #rotation = torch.cat((c, -s, s, c), dim=-1).reshape(nx*ny, d, d)

            #next_covariances = (rotation @ (torch.diag_embed(scale) @ new_covariances.reshape(nx*ny, d, d))).reshape(nx, ny, d, d)

            ut = gaussians.sample_gaussians(
                new_means, inv_sqrt_det, conics, opacities, deltas, samples) # 100, nx, ny, c

            derivatives = gaussians.gaussian_derivative(
                new_means, inv_sqrt_det, conics, opacities, new_u, samples) # 100, nx, ny, d, c

            derivatives2 = gaussians.gaussian_derivative2(
                new_means, inv_sqrt_det, conics, opacities, new_u, samples) # 100, nx, ny, d, d, c

            sample_u = gaussians.sample_gaussians(
                new_means, inv_sqrt_det, conics, opacities, new_u, samples) # 100, nx, ny, c

            bc_sample_u = gaussians.sample_gaussians(
                new_means, inv_sqrt_det, conics, opacities, new_u, bc_samples) # 100, nx, ny, c

            new_means = next_means
            new_scaling = next_scaling
            new_transform = next_transform
            new_u = next_u

            ut = ut.sum((1,2)) # 100, c
            ux = derivatives.sum((1,2)) # 100, d, c
            uxx = derivatives2.sum((1,2)) # 100, d, d, c
            sample_u = sample_u.sum((1,2)) # 100, c
            bc_sample_u = bc_sample_u.sum((1,2)) # 100, c

            if problem == "DIFFUSION":
                pde_loss += torch.mean((ut - (uxx[:,0,0] + uxx[:,1,1])) ** 2)
            elif problem == "BURGERS":
                pde_loss += torch.mean((ut + sample_u * ux.sum(-2) - nu * (uxx[:,0,0] + uxx[:,1,1])) ** 2)
            elif problem == "POISSON":
                x = samples[...,0]
                pde_loss += torch.mean((uxx[:,0,0] - 100.0 * t * torch.sin(np.pi * (x + 1.0))) ** 2)
            elif problem == "WAVE":
                pde_loss += torch.mean((ut[...,0] - sample_u[...,1]) ** 2)
                pde_loss += 0.1 * torch.mean((ut[...,1] - 10 * (uxx[...,0,0,0] + uxx[...,1,1,0]) + 0.1 * sample_u[...,1]) ** 2)

            bc_loss += torch.mean(bc_sample_u ** 2)
            # conservation_loss += torch.mean(translation ** 2)
            # conservation_loss += torch.mean(t ** 2)
            # conservation_loss += torch.mean(scale ** 2)

        loss = pde_loss + bc_loss# + conservation_loss

        loss.backward()
        optim.step()
        optim.zero_grad()

        if loss.item() < loss_cutoff:
            break

        if epoch % log_step == 0:
            training_error.append(loss.item())
            # print("Current dx:", model.dx.item())
            print("Epoch {}: Total Loss {}".format(epoch, training_error[-1]))
            print("  BC Loss:", bc_loss.item())
            print("  PDE Loss:", pde_loss.item())
            # print("  Conservation Loss:", conservation_loss.item())
            print("  Deltas:", torch.median(deltas).item(), torch.mean(deltas).item(), torch.min(deltas).item(), torch.max(deltas).item())

        if epoch % save_step == 0:
            torch.save(models.state_dict(), "checkpoints/{}_models_{}.pt".format(problem.lower(), start + epoch))

    plt.figure()
    plt.plot(np.linspace(0, len(training_error) * log_step, len(training_error)), training_error)
    plt.yscale("log")
    plt.savefig("training_error.png")

    torch.save(models.state_dict(), "{}_models.pt".format(problem.lower()))

covariances = gaussians.build_covariances(scaling, transform)
conics = torch.inverse(covariances)
inv_sqrt_det = inv_sqrt_pi * torch.sqrt(torch.det(conics))

res = 32
if problem == "WAVE":
    img1 = gaussians.sample_gaussians_img(
        means, inv_sqrt_det, conics, opacities, u[...,0], res, res
    ).detach().cpu().numpy()

    img2 = gaussians.sample_gaussians_img(
        means, inv_sqrt_det, conics, opacities, u[...,1], res, res
    ).detach().cpu().numpy()
else:
    img1 = gaussians.sample_gaussians_img(
        means, inv_sqrt_det, conics, opacities, u, res, res
    ).detach().cpu().numpy()

os.makedirs("results", exist_ok=True)

models.eval()
with torch.no_grad():
    new_means = means.clone()
    new_scaling = scaling.clone()
    new_transform = transform.clone()
    new_u = u.clone()

    covariances = gaussians.build_covariances(scaling, transform)
    old_conics = torch.inverse(covariances)

    imgs1 = []
    imgs2 = []
    imgs3 = []
    imgs4 = []

    vmax = -np.inf
    vmin = np.inf

    for i in range(test_timesteps):
        t = i * dt

        covariances = gaussians.build_covariances(new_scaling, new_transform)
        conics = torch.inverse(covariances)
        inv_sqrt_det = inv_sqrt_pi * torch.sqrt(torch.det(conics))

        in_samples = gaussians.sample_gaussians_region(
                new_means, inv_sqrt_det, conics, opacities, new_u, new_means, (5, 5), (dx, dx)) # nx*ny*5*5, nx, ny, c
        in_samples = in_samples.reshape(nx*ny * 5 * 5, nx*ny, channels).transpose(0, 1).reshape(nx*ny, channels, 5, 5, nx, ny).sum((-1, -2)) # nx*ny, c, 5, 5

        if problem == "WAVE":
            delta1 = model1(in_samples, new_means.reshape(nx*ny, d)).reshape(nx, ny)
            delta2 = model2(in_samples, new_means.reshape(nx*ny, d)).reshape(nx, ny)
            next_u = torch.stack((new_u[...,0] + delta1, new_u[...,1] + delta2), dim=-1)
        else:
            delta = model(in_samples, new_means.reshape(nx*ny, d)).reshape(nx, ny, 1)
            next_u = new_u + delta

        translation = translation_model(in_samples, new_means.reshape(nx*ny, d)).reshape(nx, ny, d)
        next_means = new_means + translation

        scale = scale_model(in_samples, new_means.reshape(nx*ny, d)).reshape(nx,ny,d)
        scale = torch.exp(scale)
        next_scaling = new_scaling * scale

        t = transform_model(in_samples, new_means.reshape(nx*ny, d)).reshape(nx,ny,-1)
        next_transform = new_transform + t

        if problem == "WAVE":
            img3 = gaussians.sample_gaussians_img(
                new_means, inv_sqrt_det, conics, opacities, new_u[...,0].unsqueeze(-1), res, res
            ).detach().cpu().numpy()

            img4 = gaussians.sample_gaussians_img(
                new_means, inv_sqrt_det, conics, opacities, new_u[...,1].unsqueeze(-1), res, res
            ).detach().cpu().numpy()
        else:
            img2 = gaussians.sample_gaussians_img(
                new_means, inv_sqrt_det, conics, opacities, new_u, res, res
            ).detach().cpu().numpy()

            img3 = gaussians.sample_gaussians_img(
                means, inv_sqrt_det, old_conics, opacities, new_u, res, res
            ).detach().cpu().numpy()

            img4 = gaussians.sample_gaussians_img(
                new_means, inv_sqrt_det, conics, opacities, u, res, res
            ).detach().cpu().numpy()

        imgs1.append(img1)
        imgs2.append(img2)
        imgs3.append(img3)
        imgs4.append(img4)

        new_means = next_means
        new_scaling = next_scaling
        new_transform = next_transform
        new_u = next_u

    for i in range(test_timesteps):
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
