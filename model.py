import enum

import numpy as np
import torch
import torch.nn.functional as f

from torch import nn

import gaussians

class Problem(enum.Enum):
    DIFFUSION = enum.auto()
    POISSON = enum.auto()
    BURGERS = enum.auto()
    WAVE = enum.auto()

class IntegrationRule(enum.Enum):
    TRAPEZOID = enum.auto()
    FORWARD = enum.auto()
    BACKWARD = enum.auto()

class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__() 
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

class Network(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation):
        super(Network, self).__init__()
        # self.dx = nn.Parameter(torch.ones(1, device="cuda") * 0.1)

        self.samples_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size),
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

class Model(nn.Module):
    def __init__(self, problem, nx, ny, d, dx, dt, kernel_size, rule):
        super(Model, self).__init__()
        self.problem = problem
        self.nx = nx
        self.ny = ny
        self.d = d
        self.dx = dx
        self.dt = dt
        self.kernel_size = kernel_size
        self.rule = rule

        self.pde_weight = 1
        self.bc_weight = 1
        self.conservation_weight = 1

        self.channels = 2 if problem == Problem.WAVE else 1

        tx = torch.linspace(-1, 1, nx).cuda()
        ty = torch.linspace(-1, 1, ny).cuda()
        gx, gy = torch.meshgrid((tx,ty), indexing="xy")
        self.initial_means = torch.stack((gx,gy), dim=-1)
        scaling = torch.ones((nx,ny,d), device="cuda") * -4.0
        self.initial_scaling = torch.exp(scaling)
        transform = torch.zeros((nx,ny,d * (d - 1) // 2), device="cuda")
        self.initial_transform = f.tanh(transform)
        self.initial_opacities = torch.ones((nx,ny), device="cuda") * 0.5

        self.inv_sqrt_pi = np.power(1.0 / np.sqrt(2.0 * np.pi), d)

        self.covariances = gaussians.build_covariances(self.initial_scaling, self.initial_transform)
        self.initial_conics = torch.inverse(self.covariances)
        self.initial_inv_sqrt_det = self.inv_sqrt_pi * torch.sqrt(torch.det(self.initial_conics))

        if problem == Problem.POISSON:
            self.initial_u = torch.zeros((nx, ny), device="cuda")
            # u = torch.sin(np.pi * (means[...,0] + 1.0))
            # u = torch.sin(np.pi * (means[...,0] + 1.0) * (means[...,1] + 1.0))
        elif problem == Problem.WAVE:
            self.initial_u = torch.zeros((nx, ny), device="cuda")
            self.initial_u[nx//2-1,ny//2-1] = 1.0
            self.initial_u[nx//2-1,ny//2] = 1.0
            self.initial_u[nx//2,ny//2-1] = 1.0
            self.initial_u[nx//2,ny//2] = 1.0
        else:
            sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, 1, d, 1)
            samples = self.initial_means.unsqueeze(-1) - sample_mean
            conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1)
            powers = -0.5 * (samples.transpose(-1, -2) @ (conics @ samples))
            self.initial_u = torch.exp(powers).squeeze()
            self.initial_u = self.initial_u / torch.max(self.initial_u)

        self.initial_u = self.initial_u.unsqueeze(-1).repeat(1, 1, self.channels)

        self.translation_model = Network(self.channels, d, kernel_size, nn.SiLU()).cuda()
        self.scale_model = Network(self.channels, d, kernel_size, nn.SiLU()).cuda()
        self.transform_model = Network(self.channels, transform.shape[-1], kernel_size, nn.SiLU()).cuda()

        if problem == Problem.WAVE:
            self.model1 = Network(self.channels, 1, kernel_size, WaveAct()).cuda()
            self.model2 = Network(self.channels, 1, kernel_size, WaveAct()).cuda()
        else:
            self.model = Network(self.channels, 1, kernel_size, WaveAct()).cuda()

        self.reset()

    def reset(self):
        self.u = self.initial_u
        self.means = self.initial_means
        self.scaling = self.initial_scaling
        self.transform = self.initial_transform
        self.opacities = self.initial_opacities
        self.inv_sqrt_det = self.initial_inv_sqrt_det
        self.conics = self.initial_conics
        self.deltas = torch.zeros((self.nx, self.ny, self.channels), device="cuda")

        self.ut_samples = []
        self.ux_samples = []
        self.uxx_samples = []
        self.u_samples = []
        self.bc_u_samples = []

    def forward(self, dt):
        in_samples = gaussians.sample_gaussians_region(
            self.means, self.inv_sqrt_det, self.conics, self.initial_opacities, self.u,
            self.means, (self.kernel_size, self.kernel_size), (self.dx, self.dx)
        ) # nx*ny*k*k, nx, ny, c
        in_samples = in_samples.reshape(
            self.nx * self.ny * self.kernel_size * self.kernel_size, self.nx * self.ny, self.channels
        ).transpose(0, 1).reshape(
            self.nx * self.ny, self.channels, self.kernel_size, self.kernel_size, self.nx, self.ny
        ).sum((-1, -2)) # nx*ny, c, k, k

        if self.problem == Problem.WAVE:
            delta1 = self.model1(
                in_samples, self.means.reshape(self.nx * self.ny, self.d)).reshape(self.nx, self.ny)
            delta2 = self.model2(
                in_samples, self.means.reshape(self.nx * self.ny, self.d)).reshape(self.nx, self.ny)
            self.deltas = torch.stack((delta1, delta2), dim=-1) / self.dt
            next_u = torch.stack((self.u[...,0] + delta1, self.u[...,1] + delta2), dim=-1)
        else:
            delta = self.model(
                in_samples, self.means.reshape(self.nx * self.ny, self.d)).reshape(self.nx, self.ny, 1)
            self.deltas = delta / self.dt
            next_u = self.u + delta

        self.translation = self.translation_model(
            in_samples, self.means.reshape(self.nx * self.ny, self.d)
        ).reshape(self.nx, self.ny, self.d)
        next_means = self.means + self.translation

        self.dscale = self.scale_model(
            in_samples, self.means.reshape(self.nx * self.ny, self.d)
        ).reshape(self.nx, self.ny, self.d)
        scale = torch.exp(self.dscale)
        next_scaling = self.scaling * scale

        self.dtransform = self.transform_model(
            in_samples, self.means.reshape(self.nx * self.ny, self.d)
        ).reshape(self.nx, self.ny, -1)
        next_transform = self.transform + self.dtransform

        self.means = next_means
        self.scaling = next_scaling
        self.transform = next_transform
        self.u = next_u


    def sample(self, samples, bc_samples):
        self.covariances = gaussians.build_covariances(self.scaling, self.transform)
        self.conics = torch.inverse(self.covariances)
        self.inv_sqrt_det = self.inv_sqrt_pi * torch.sqrt(torch.det(self.conics))

        ut = gaussians.sample_gaussians(
            self.means, self.inv_sqrt_det, self.conics, self.opacities, self.deltas, samples
        ) # 100, nx, ny, c
        ux = gaussians.gaussian_derivative(
            self.means, self.inv_sqrt_det, self.conics, self.opacities, self.u, samples
        ) # 100, nx, ny, d, c
        uxx = gaussians.gaussian_derivative2(
            self.means, self.inv_sqrt_det, self.conics, self.opacities, self.u, samples
        ) # 100, nx, ny, d, d, c
        u_sample = gaussians.sample_gaussians(
            self.means, self.inv_sqrt_det, self.conics, self.opacities, self.u, samples
        ) # 100, nx, ny, c
        bc_u_sample = gaussians.sample_gaussians(
            self.means, self.inv_sqrt_det, self.conics, self.opacities, self.u, bc_samples
        ) # 100, nx, ny, c

        ut = ut.sum((1,2)) # 100, c
        ux = ux.sum((1,2)) # 100, d, c
        uxx = uxx.sum((1,2)) # 100, d, d, c
        u_sample = u_sample.sum((1,2)) # 100, c
        bc_u_sample = bc_u_sample.sum((1,2)) # 100, c

        self.ut_samples.append(ut)
        self.ux_samples.append(ux)
        self.uxx_samples.append(uxx)
        self.u_samples.append(u_sample)
        self.bc_u_samples.append(bc_u_sample)

    def compute_loss(self, t, samples, bc_samples):
        self.sample(samples, bc_samples)

        if self.rule == IntegrationRule.TRAPEZOID:
            ux = 0.5 * (self.ux_samples[-1] + self.ux_samples[-2])
            uxx = 0.5 * (self.uxx_samples[-1] + self.uxx_samples[-2])
            u_sample = 0.5 * (self.u_samples[-1] + self.u_samples[-2])
        elif self.rule == IntegrationRule.FORWARD:
            ux = self.ux_samples[-2]
            uxx = self.uxx_samples[-2]
            u_sample = self.u_samples[-2]
        elif self.rule == IntegrationRule.BACKWARD:
            ux = self.ux_samples[-1]
            uxx = self.uxx_samples[-1]
            u_sample = self.u_samples[-1]
        else:
            raise ValueError("Unexpected integration rule:", self.rule)

        ut = self.ut_samples[-1]
        bc_u_sample = self.bc_u_samples[-1]

        pde_loss = torch.zeros(1, device="cuda")
        bc_loss = torch.zeros(1, device="cuda")
        conservation_loss = torch.zeros(1, device="cuda")

        if self.problem == Problem.DIFFUSION:
            pde_loss += torch.mean((ut - (uxx[:,0,0] + uxx[:,1,1])) ** 2)
        elif self.problem == Problem.BURGERS:
            pde_loss += torch.mean((ut + u_sample * ux.sum(-2) - nu * (uxx[:,0,0] + uxx[:,1,1])) ** 2)
        elif self.problem == Problem.POISSON:
            x = samples[...,0]
            pde_loss += torch.mean((uxx[:,0,0] - 100.0 * t * torch.sin(np.pi * (x + 1.0))) ** 2)
        elif self.problem == Problem.WAVE:
            pde_loss += torch.mean((ut[...,0] - u_sample[...,1]) ** 2)
            pde_loss += 0.1 * torch.mean(
                (ut[...,1] - 10 * (uxx[...,0,0,0] + uxx[...,1,1,0]) + 0.1 * u_sample[...,1]) ** 2)
        else:
            raise ValueError("Unexpected PDE problem:", self.problem)

        bc_loss += torch.mean(bc_u_sample ** 2)
        conservation_loss += torch.mean(self.translation ** 2)
        conservation_loss += torch.mean(self.dtransform ** 2)
        conservation_loss += torch.mean(self.dscale ** 2)

        return self.pde_weight * pde_loss,\
               self.bc_weight * bc_loss,\
               self.conservation_weight * conservation_loss

    def generate_images(self, res):
        if self.problem == Problem.WAVE:
            img1 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_inv_sqrt_det, self.initial_conics,
                self.initial_opacities, self.initial_u[...,0], res, res
            ).detach().cpu().numpy()

            img2 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_inv_sqrt_det, self.initial_conics,
                self.initial_opacities, self.initial_u[...,1], res, res
            ).detach().cpu().numpy()

            img3 = gaussians.sample_gaussians_img(
                self.means, self.inv_sqrt_det, self.conics,
                self.initial_opacities, self.u[...,0].unsqueeze(-1), res, res
            ).detach().cpu().numpy()

            img4 = gaussians.sample_gaussians_img(
                self.means, self.inv_sqrt_det, self.conics,
                self.initial_opacities, self.u[...,1].unsqueeze(-1), res, res
            ).detach().cpu().numpy()
        else:
            img1 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_inv_sqrt_det, self.initial_conics,
                self.initial_opacities, self.initial_u, res, res
            ).detach().cpu().numpy()

            img2 = gaussians.sample_gaussians_img(
                self.means, self.inv_sqrt_det, self.conics,
                self.initial_opacities, self.u, res, res
            ).detach().cpu().numpy()

            img3 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_inv_sqrt_det, self.initial_conics,
                self.initial_opacities, self.u, res, res
            ).detach().cpu().numpy()

            img4 = gaussians.sample_gaussians_img(
                self.means, self.inv_sqrt_det, self.conics,
                self.initial_opacities, self.initial_u, res, res
            ).detach().cpu().numpy()

        return img1, img2, img3, img4

    def plot_gaussians(self):
        return gaussians.plot_gaussians(self.means, self.covariances, self.opacities, self.u)
