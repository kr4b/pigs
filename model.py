import enum

import numpy as np
import torch
import torch.nn.functional as f

from torch import nn

import gaussians

from diff_gaussian_sampling import GaussianSampler

class Problem(enum.Enum):
    DIFFUSION = enum.auto()
    POISSON = enum.auto()
    BURGERS = enum.auto()
    WAVE = enum.auto()
    NAVIER_STOKES = enum.auto()

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
    def __init__(self, problem, nx, ny, d, dx, dt, kernel_size, rule, train_opacity, scale):
        super(Model, self).__init__()
        self.problem = problem
        self.nx = nx
        self.ny = ny
        self.d = d
        self.dx = dx
        self.dt = dt
        self.kernel_size = kernel_size
        self.rule = rule
        self.train_opacity = train_opacity
        self.scale = scale

        self.nu = 1.0 / (100.0 * np.pi)
        self.inv_rho = 0.1
        self.mu = 0.1

        self.pde_weight = 1
        self.bc_weight = 1
        self.conservation_weight = 1

        self.channels = 1

        tx = torch.linspace(-1, 1, nx).cuda()
        ty = torch.linspace(-1, 1, ny).cuda()
        gx, gy = torch.meshgrid((tx,ty), indexing="ij")
        self.initial_means = torch.stack((gx,gy), dim=-1)
        self.initial_means = self.initial_means.reshape(-1, d)
        scaling = torch.ones((nx*ny,d), device="cuda") * -5.5
        # scaling[10*ny+20] = -3.5
        # scaling[8*ny+20] = -3.5
        # scaling[20*ny+20] = -3.5
        # scaling[22*ny+20] = -3.5

        self.initial_scaling = torch.exp(scaling)
        self.transform_size = d * (d - 1) // 2
        self.initial_transform = torch.zeros((nx*ny,self.transform_size), device="cuda")
        self.initial_opacities = torch.sigmoid(torch.zeros((nx*ny), device="cuda"))

        self.covariances = gaussians.build_covariances(self.initial_scaling, self.initial_transform)
        self.initial_conics = torch.inverse(self.covariances)

        if problem == Problem.POISSON:
            self.initial_u = torch.zeros((nx*ny), device="cuda")
        elif problem == Problem.WAVE:
            self.channels = 2
            self.initial_u = torch.zeros((nx*ny, self.channels), device="cuda")
            self.initial_u[(nx//2-1)*ny+ny//2-1,1] = 1.0
            self.initial_u[(nx//2-1)*ny+ny//2,1] = 1.0
            self.initial_u[(nx//2)*ny+ny//2-1,1] = 1.0
            self.initial_u[(nx//2)*ny+ny//2,1] = 1.0
        elif problem == Problem.NAVIER_STOKES:
            self.channels = 3
            self.initial_u = torch.zeros((nx*ny, self.channels), device="cuda")
            self.initial_u[1*ny+1:1*ny+ny-2,0] = 1.0
        else:
            sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, d, 1)
            samples = self.initial_means.unsqueeze(-1) - sample_mean
            conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1)
            powers = -0.5 * (samples.transpose(-1, -2) @ (conics @ samples))
            self.initial_u = torch.exp(powers).squeeze()
            self.initial_u = self.initial_u

        # self.initial_u = torch.zeros((nx,ny), device="cuda")

        # for i in range(nx):
        #     for j in range(ny):
        #         x = i / 15 - 1.0
        #         y = j / 15 - 1.0
        #         if y < -0.2 and 0.3 < x**2 + y**2 < 0.6:
        #             self.initial_u[i,j] = 1.0

        # self.initial_u[10,20] = 3.0
        # self.initial_u[20,20] = 3.0
        # self.initial_u[9,20] = 2.0
        # self.initial_u[8,20] = 3.0
        # self.initial_u[21,20] = 2.0
        # self.initial_u[22,20] = 3.0

        self.initial_u = self.initial_u.reshape(nx*ny, self.channels) * 0.25

        out_channels = d + d + self.transform_size

        if self.train_opacity:
            out_channels += 1

        self.gaussian_model = Network(
            self.channels, out_channels, self.kernel_size, nn.SiLU()).cuda()

        if self.problem == Problem.NAVIER_STOKES:
            self.solution_model_v = Network(
                self.channels, 2, self.kernel_size, nn.Tanh()).cuda()
            self.solution_model_p = Network(
                self.channels, 1, self.kernel_size, nn.Tanh()).cuda()
        else:
            self.solution_model = Network(
                self.channels, self.channels, self.kernel_size, nn.Tanh()).cuda()

        self.sampler = GaussianSampler(False)
        self.sample_kernel = \
            gaussians.region_kernel(self.kernel_size, self.dx, self.d).reshape(1, -1, d)

        self.reset()

    def reset(self):
        self.u = self.initial_u
        self.means = self.initial_means
        self.scaling = self.initial_scaling
        self.transform = self.initial_transform
        self.opacities = self.initial_opacities
        self.raw_opacities = torch.zeros(self.opacities.shape, device="cuda")
        self.covariances = gaussians.build_covariances(self.scaling, self.transform)
        self.conics = self.initial_conics

        self.clear()

    def clear(self):
        # self.ut_samples = []
        self.ux_samples = []
        self.uxx_samples = []
        self.u_samples = []
        self.bc_u_samples = []

    def detach(self):
        self.u = self.u.detach()
        self.means = self.means.detach()
        self.scaling = self.scaling.detach()
        self.transform = self.transform.detach()
        self.opacities = self.opacities.detach()
        self.raw_opacities = self.raw_opacities.detach()
        self.covariances = self.covariances.detach()
        self.conics = self.conics.detach()

    def bc_mask(self, samples):
        mask = torch.any(torch.abs(samples) < self.scale, -1).reshape(-1, 1)
        if self.problem == Problem.NAVIER_STOKES:
            mask = torch.logical_or(mask,
                torch.any(torch.abs(samples) > 0.1, -1).reshape(-1, 1))

        return mask

    def forward(self):
        with torch.no_grad():
            samples = (
                self.means.reshape(self.nx*self.ny, 1, self.d) + self.sample_kernel
            ).reshape(-1, self.d)
            self.sampler.preprocess(
                self.means, self.u, self.covariances, self.conics, self.opacities, samples)
            in_samples = self.sampler.sample_gaussians() # nx*ny*k*k, c

        in_samples = in_samples.reshape(
            self.nx * self.ny, self.kernel_size * self.kernel_size, self.channels
        ).transpose(-1, -2).reshape(
            self.nx * self.ny, self.channels, self.kernel_size, self.kernel_size
        ) # nx*ny, c, k, k

        if self.problem == Problem.NAVIER_STOKES:
            delta_v = self.solution_model_v(in_samples, self.means).reshape(self.nx*self.ny, 2)
            delta_p = self.solution_model_p(in_samples, self.means).reshape(self.nx*self.ny, 1)

            deltas = torch.cat((delta_v, delta_p), dim=-1)
        else:
            deltas = self.solution_model(in_samples, self.means).reshape(self.nx*self.ny, self.channels)

        self.u = self.u + deltas

        z = self.gaussian_model(
            in_samples, self.means.reshape(self.nx * self.ny, self.d)
        ).reshape(self.nx*self.ny, -1)

        self.translation = z[...,:self.d]
        self.means = self.means + self.translation

        self.dscale = z[...,self.d:2*self.d]
        scale = torch.exp(self.dscale)
        self.scaling = self.scaling * scale

        self.dtransform = z[...,2*self.d:2*self.d + self.transform_size]
        self.transform = self.transform + self.dtransform

        if self.train_opacity:
            self.dopacities = z[...,2*self.d + self.transform_size]
            self.raw_opacities = self.raw_opacities + self.dopacities
            self.opacities = torch.sigmoid(self.raw_opacities)

        self.covariances = gaussians.build_covariances(self.scaling, self.transform)
        self.conics = torch.inverse(self.covariances)

    def sample(self, samples, bc_samples):
        # u_sample = gaussians.sample_gaussians(
        #     self.means, self.conics, self.opacities, self.u, samples
        # ) # n, c
        # bc_u_sample = gaussians.sample_gaussians(
        #     self.means, self.conics, self.opacities, self.u, bc_samples
        # ) # n, c

        # ux = torch.autograd.grad(u_sample.sum(), samples, retain_graph=True, create_graph=True)[0]
        # uxx_x = torch.autograd.grad(ux[:,0].sum(), samples, retain_graph=True)[0].unsqueeze(-1)
        # uxx_y = torch.autograd.grad(ux[:,1].sum(), samples)[0].unsqueeze(-1)
        # uxx = torch.cat((uxx_x, uxx_y), dim=-1)
        # ux = gaussians.gaussian_derivative(
        #     self.means, self.conics, self.opacities, self.u, samples
        # ) # n, d, c
        # uxx = gaussians.gaussian_derivative2(
        #     self.means, self.conics, self.opacities, self.u, samples
        # ) # n, d, d, c

        self.sampler.preprocess(
            self.means, self.u, self.covariances, self.conics, self.opacities, samples)
        u_sample = self.sampler.sample_gaussians() # n, c
        ux = self.sampler.sample_gaussians_derivative() # n, d, c
        uxx = self.sampler.sample_gaussians_laplacian() # n, d, d, c

        self.sampler.preprocess(
            self.means, self.u, self.covariances, self.conics, self.opacities, bc_samples)
        bc_u_sample = self.sampler.sample_gaussians() # n, c

        self.u_samples.append(u_sample)
        self.bc_u_samples.append(bc_u_sample)
        self.ux_samples.append(ux)
        self.uxx_samples.append(uxx)

    def compute_loss(self, t, samples, time_samples, bc_samples):
        self.sample(samples, bc_samples)

        if self.rule == IntegrationRule.TRAPEZOID:
            ux = time_samples.reshape(-1, 1, 1) * self.ux_samples[-1] \
               + (1 - time_samples.reshape(-1, 1, 1)) * self.ux_samples[-2]
            uxx = time_samples.reshape(-1, 1, 1, 1) * self.uxx_samples[-1] \
                + (1 - time_samples.reshape(-1, 1, 1, 1)) * self.uxx_samples[-2]
            u_sample = time_samples.reshape(-1, 1) * self.u_samples[-1] \
                     + (1 - time_samples.reshape(-1, 1)) * self.u_samples[-2]
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

        ut = (self.u_samples[-1] - self.u_samples[-2]) / self.dt
        bc_u_sample = self.bc_u_samples[-1]

        pde_loss = torch.zeros(1, device="cuda")
        bc_loss = torch.zeros(1, device="cuda")
        conservation_loss = torch.zeros(1, device="cuda")

        if self.problem == Problem.DIFFUSION:
            pde_loss += torch.mean((ut - (uxx[:,0,0] + uxx[:,1,1])) ** 2)

        elif self.problem == Problem.BURGERS:
            pde_loss += \
                torch.mean((ut + u_sample * ux[:,0] - self.nu * (uxx[:,0,0] + uxx[:,1,1])) ** 2)

        elif self.problem == Problem.POISSON:
            x = samples[...,0]
            pde_loss += torch.mean((uxx[:,0,0] - 100.0 * t * torch.sin(np.pi * (x + 1.0))) ** 2)

        elif self.problem == Problem.WAVE:
            pde_loss += torch.mean((ut[...,0] - u_sample[...,1]) ** 2)
            pde_loss += 0.01 * torch.mean(
                (ut[...,1] - 10 * (uxx[...,0,0,0] + uxx[...,1,1,0]) + 0.1 * u_sample[...,1]) ** 2)

        elif self.problem == Problem.NAVIER_STOKES:
            bc_mask = self.bc_mask(samples)
            pde_loss += torch.mean(bc_mask * (ux[...,:2].sum(-1)) ** 2)
            pde_loss += torch.mean(bc_mask * (
                ut[...,:2] + (u_sample[...,:2].reshape(-1, 1, 2) * ux[...,:2]).sum(-2) \
                + self.inv_rho * ux[...,-1] \
                - self.mu * (uxx[:,0,0,:2] + uxx[:,1,1,:2])
            ) ** 2)

        else:
            raise ValueError("Unexpected PDE problem:", self.problem)

        bc_loss += torch.mean(bc_u_sample ** 2)
        conservation_loss += torch.mean(self.translation ** 2)
        conservation_loss += torch.mean(self.dtransform ** 2)
        conservation_loss += torch.mean(self.dscale ** 2)
        if self.train_opacity:
            conservation_loss += torch.mean(self.dopacities ** 2)

        return self.pde_weight * pde_loss,\
               self.bc_weight * bc_loss,\
               self.conservation_weight * conservation_loss

    def generate_images(self, res, scale):
        if self.problem == Problem.WAVE:
            img1 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics,
                self.initial_opacities, self.initial_u[...,0], res, res, scale
            ).detach().cpu().numpy()

            img2 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics,
                self.initial_opacities, self.initial_u[...,1], res, res, scale
            ).detach().cpu().numpy()

            img3 = gaussians.sample_gaussians_img(
                self.means, self.conics,
                self.opacities, self.u[...,0], res, res, scale
            ).detach().cpu().numpy()

            img4 = gaussians.sample_gaussians_img(
                self.means, self.conics,
                self.opacities, self.u[...,1], res, res, scale
            ).detach().cpu().numpy()
        if self.problem == Problem.NAVIER_STOKES:
            img1 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics,
                self.initial_opacities, torch.sqrt((self.initial_u[...,:2] ** 2).sum(-1)), res, res, scale
            ).detach().cpu().numpy()

            img2 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics,
                self.initial_opacities, self.initial_u[...,-1], res, res, scale
            ).detach().cpu().numpy()

            img3 = gaussians.sample_gaussians_img(
                self.means, self.conics,
                self.opacities, torch.sqrt((self.u[...,:2] ** 2).sum(-1)), res, res, scale
            ).detach().cpu().numpy()

            img4 = gaussians.sample_gaussians_img(
                self.means, self.conics,
                self.opacities, self.u[...,-1], res, res, scale
            ).detach().cpu().numpy()
        else:
            img1 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics,
                self.initial_opacities, self.initial_u, res, res, scale
            ).detach().cpu().numpy()

            img2 = gaussians.sample_gaussians_img(
                self.means, self.conics,
                self.opacities, self.u, res, res, scale
            ).detach().cpu().numpy()

            img3 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics,
                self.opacities, self.u, res, res, scale
            ).detach().cpu().numpy()

            img4 = gaussians.sample_gaussians_img(
                self.means, self.conics,
                self.initial_opacities, self.initial_u, res, res, scale
            ).detach().cpu().numpy()

        return img1, img2, img3, img4

    def plot_gaussians(self):
        return gaussians.plot_gaussians(self.means, self.covariances, self.opacities, self.u)
