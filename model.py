import time
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
    def __init__(self, in_channels, out_channels, c, d, kernel_size, activation):
        super(Network, self).__init__()
        # self.dx = nn.Parameter(torch.ones(1, device="cuda") * 0.1)

        self.samples_conv = nn.Sequential(
            nn.Conv2d(in_channels, 50, kernel_size),
            # nn.LayerNorm((64, 1, 1)),
            activation,
            nn.Flatten(),
        )
        self.param_embed = nn.Sequential(
            nn.Linear(d + d + d * (d - 1) // 2 + c + 1, 30),
            # nn.LayerNorm(64),
            activation,
        )
        self.linear = nn.Sequential(
            nn.Linear(80, 80),
            # nn.LayerNorm(256),
            activation,
            # nn.Linear(256, 256),
            # nn.LayerNorm(256),
            # activation,
            nn.Linear(80, out_channels),
        )

    def forward(self, img, x):
        y = self.samples_conv(img)
        # embed = self.pos_embed(torch.cat(
        #     (x, torch.sin(np.pi * 2 * x), torch.cos(np.pi * 2 * x), torch.sin(np.pi * 4 * x),
        #      torch.cos(np.pi * 4 * x), torch.sin(np.pi * 6 * x), torch.cos(np.pi * 6 * x),
        #      torch.sin(np.pi * 8 * x), torch.cos(np.pi * 8 * x), torch.sin(np.pi * 10 * x),
        #      torch.cos(np.pi * 10 * x)), dim=-1)
        # )
        embed = self.param_embed(x)
        y = self.linear(torch.cat((y, embed), dim=1))
        # y = self.linear(y)
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

        self.pde_weight = 1.0
        self.bc_weight = 1.0
        self.conservation_weight = 1.0

        self.channels = 1

        tx = torch.linspace(-1, 1, nx).cuda() * self.scale
        ty = torch.linspace(-1, 1, ny).cuda() * self.scale
        gx, gy = torch.meshgrid((tx,ty), indexing="ij")
        self.initial_means = torch.stack((gx,gy), dim=-1)
        self.initial_means = self.initial_means.reshape(-1, d)
        scaling = torch.ones((nx*ny,d), device="cuda") * -4.5
        # scaling[10*ny+20] = -3.5
        # scaling[8*ny+20] = -3.5
        # scaling[20*ny+20] = -3.5
        # scaling[22*ny+20] = -3.5

        self.initial_scaling = torch.exp(scaling) * self.scale
        self.transform_size = d * (d - 1) // 2
        self.initial_transform = torch.zeros((nx*ny,self.transform_size), device="cuda")
        if self.train_opacity:
            self.initial_opacities = torch.zeros((nx*ny), device="cuda")
        else:
            self.initial_opacities = torch.ones((nx*ny), device="cuda") * 100.0

        if problem == Problem.POISSON:
            self.initial_u = torch.zeros((nx*ny), device="cuda")
        elif problem == Problem.WAVE:
            self.channels = 2
            self.initial_u = torch.zeros((nx*ny, self.channels), device="cuda")
            for i in range(4):
                for j in range(4):
                    self.initial_u[(nx//2-3+i)*ny+ny//2-3+j,1] = 0.25
        elif problem == Problem.NAVIER_STOKES:
            self.channels = 3
            self.initial_u = torch.cat((
                torch.ones((ny, 1), device="cuda") * 0.25,
                torch.zeros((ny, 2), device="cuda"),
            ), dim=-1)
            self.initial_scaling = self.initial_scaling[ny:2*ny]
            self.initial_means = self.initial_means[ny:2*ny]
            self.initial_transform = self.initial_transform[ny:2*ny]
            self.initial_opacities = self.initial_opacities[ny:2*ny]
            # self.initial_u[2*ny+2:2*ny+ny-3,0] = 1.0
            # self.initial_u[3*ny+2:3*ny+ny-3,0] = 1.0
        else:
            sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, d, 1)
            samples = self.initial_means.unsqueeze(-1) - sample_mean
            conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1 * self.scale)
            powers = -0.5 * (samples.transpose(-1, -2) @ (conics @ samples))
            self.initial_u = torch.exp(powers).squeeze(-1)

        self.initial_covariances = gaussians.build_covariances(self.initial_scaling, self.initial_transform)
        self.initial_conics = torch.inverse(self.initial_covariances)

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

        # self.initial_u = self.initial_u.reshape(nx*ny, self.channels) * 0.25

        # out_channels = d + d + self.transform_size

        # if self.train_opacity:
        #     out_channels += 1

        in_channels = self.channels + 2

        if self.problem == Problem.NAVIER_STOKES:
            in_channels += 1
            self.solution_model_v = Network(
                in_channels, 2, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
            self.solution_model_p = Network(
                in_channels, 1, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
        else:
            self.solution_model = Network(
                in_channels, self.channels, self.channels, d, self.kernel_size, nn.Tanh()).cuda()

        self.translation_model = Network(
            in_channels, d, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
        self.scale_model = Network(
            in_channels, d, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
        self.transform_model = Network(
            in_channels, self.transform_size, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
        if self.train_opacity:
            self.opacity_model = Network(
                in_channels, 1, self.channels, d, self.kernel_size, nn.Tanh()).cuda()

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
        self.covariances = self.initial_covariances
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
        self.covariances = self.covariances.detach()
        self.conics = self.conics.detach()
        self.translation = self.translation.detach()
        self.dscale = self.dscale.detach()
        self.dtransform = self.dtransform.detach()
        if self.train_opacity:
            self.dopacities = self.dopacities.detach()

    def bc_mask(self, samples):
        mask = torch.all(torch.abs(samples) < self.scale, -1).reshape(-1, 1)
        if self.problem == Problem.NAVIER_STOKES:
            # Circle
            mask = torch.logical_and(mask,
                (((samples / self.scale + torch.tensor([[0.65, 0.0]], device="cuda")) ** 2)
                .sum(-1) > 0.1 ** 2).reshape(-1, 1))
            # Square
            # mask = torch.logical_and(mask,
            #     torch.any(torch.abs(
            #         samples / self.scale + torch.tensor([[0.65, 0.0]], device="cuda")
            #     ) > 0.1, -1).reshape(-1, 1))

        return mask
    
    def pde_rhs(self, samples, u, ux, uxx):
        if self.problem == Problem.DIFFUSION:
            return uxx[:,0,0] + uxx[:,1,1]

        elif self.problem == Problem.BURGERS:
            return self.nu * (uxx[:,0,0] + uxx[:,1,1]) - u * ux[:,0]

        elif self.problem == Problem.POISSON:
            x = samples[...,0]
            return 100.0 * t * torch.sin(np.pi * (x + 1.0)) - uxx[:,0,0]

        elif self.problem == Problem.WAVE:
            return 10 * (uxx[...,0,0,0] + uxx[...,1,1,0]) - 0.1 * u[...,1]

        elif self.problem == Problem.NAVIER_STOKES:
            return self.mu * (uxx[:,0,0,:2] + uxx[:,1,1,:2]) \
                 - (u[:,:2].reshape(-1, 1, 2) * ux[...,:2]).sum(-1) \
                 - self.inv_rho * ux[...,-1]

        else:
            raise ValueError("Unexpected PDE problem:", self.problem)

    def forward(self):
        self.old_means = self.means
        self.old_covariances = self.covariances
        self.old_conics = self.conics
        self.old_opacities = self.opacities

        with torch.no_grad():
            samples = (
                self.means.reshape(self.means.shape[0], 1, self.d) + self.sample_kernel
            ).reshape(-1, self.d) # nx*ny*k*k, d
            bc_mask = self.bc_mask(samples) # nx*ny*k*k, 1
            self.sampler.preprocess(
                self.means, self.u, self.covariances, self.conics,
                torch.sigmoid(self.opacities), samples)
            u = self.sampler.sample_gaussians() # nx*ny*k*k, c
            ux = self.sampler.sample_gaussians_derivative() # nx*ny*k*k, d, c
            uxx = self.sampler.sample_gaussians_laplacian() # nx*ny*k*k, d, d, c
            pde = self.pde_rhs(samples, u, ux, uxx).reshape(u.shape[0], -1) # nx*ny*k*k, -1
            in_samples = torch.cat((u, pde, bc_mask), -1)

        in_samples = in_samples.reshape(
            self.means.shape[0], self.kernel_size * self.kernel_size, -1
        ).transpose(-1, -2).reshape(
            self.means.shape[0], -1, self.kernel_size, self.kernel_size
        ) # nx*ny, -1, k, k

        in_params = torch.cat((
            self.means.reshape(-1, 2),
            self.u.reshape(-1, self.channels),
            self.scaling.reshape(-1, 2),
            self.transform.reshape(-1, self.transform_size),
            self.opacities.reshape(-1, 1),
        ), dim=-1)
        if self.problem == Problem.NAVIER_STOKES:
            delta_v = self.solution_model_v(in_samples, in_params).reshape(self.means.shape[0], 2)
            delta_p = self.solution_model_p(in_samples, in_params).reshape(self.means.shape[0], 1)

            deltas = torch.cat((delta_v, delta_p), dim=-1)
        else:
            deltas = self.solution_model(in_samples, in_params).reshape(self.means.shape[0], self.channels)

        u = self.u + deltas

        self.translation = \
            self.translation_model(in_samples, in_params).reshape(self.means.shape[0], -1)
        means = self.means + self.translation

        self.dscale = self.scale_model(in_samples, in_params).reshape(self.means.shape[0], -1)
        scale = torch.exp(self.dscale)
        scaling = self.scaling * scale

        self.dtransform = \
            self.transform_model(in_samples, in_params).reshape(self.means.shape[0], -1)
        transform = self.transform + self.dtransform

        if self.train_opacity:
            self.dopacities = \
                self.opacity_model(in_samples, in_params).reshape(self.means.shape[0])
            opacities = self.opacities + self.dopacities
        else:
            opacities = self.opacities

        covariances = gaussians.build_covariances(scaling, transform)

        # self.u[torch.logical_and(
        #     torch.logical_and(self.means[...,0] < -1.0 + 4.0/self.nx,
        #                       self.means[...,0] > -1.0 + 2.0/self.nx),
        #             torch.abs(self.means[...,1]) < 1.0 - 2.0/self.nx), 0] = 0.25

        if self.problem == Problem.NAVIER_STOKES:
            keep_mask = torch.logical_and(
                torch.logical_and(
                    means[...,0] < self.scale,
                    torch.logical_and(torch.all(scaling < 1.0, -1), torch.all(scaling > 0.001, -1))
                ),
                torch.logical_not(torch.any(torch.abs(transform) > 3.0, -1))
            )
            self.u = torch.cat((u[keep_mask], self.initial_u), dim=0)
            self.means = torch.cat((means[keep_mask], self.initial_means), dim=0)
            self.scaling = torch.cat((scaling[keep_mask], self.initial_scaling), dim=0)
            self.transform = torch.cat((transform[keep_mask], self.initial_transform), dim=0)
            self.opacities = torch.cat((opacities[keep_mask], self.initial_opacities), dim=0)
            self.covariances = torch.cat((covariances[keep_mask], self.initial_covariances), dim=0)
            conics = torch.inverse(covariances[keep_mask])
            self.conics = torch.cat((conics, self.initial_conics), dim=0)
        else:
            # keep_mask = torch.logical_and(
            #     torch.all(scaling < 1.0, -1),
            #     torch.all(scaling > 0.001, -1)
            # )
            self.u = u
            self.means = means
            self.scaling = scaling
            self.transform = transform
            self.opacities = opacities
            self.covariances = covariances
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
            self.means, self.u, self.covariances, self.conics,
            torch.sigmoid(self.opacities), samples)
        u_sample = self.sampler.sample_gaussians() # n, c
        ux = self.sampler.sample_gaussians_derivative() # n, d, c
        uxx = self.sampler.sample_gaussians_laplacian() # n, d, d, c

        self.sampler.preprocess(
            self.means, self.u, self.covariances, self.conics,
            torch.sigmoid(self.opacities), bc_samples)
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

        rhs = self.pde_rhs(samples, u_sample, ux, uxx)

        if self.problem == Problem.DIFFUSION:
            pde_loss += torch.mean((ut - rhs) ** 2)

        elif self.problem == Problem.BURGERS:
            pde_loss += torch.mean((ut - rhs) ** 2)

        elif self.problem == Problem.POISSON:
            pde_loss += torch.mean(rhs ** 2)

        elif self.problem == Problem.WAVE:
            pde_loss += torch.mean((ut[...,0] - u_sample[...,1]) ** 2)
            pde_loss += 0.01 * torch.mean((ut[...,1] - rhs) ** 2)

        elif self.problem == Problem.NAVIER_STOKES:
            self.sampler.preprocess(
                self.old_means, self.translation, self.old_covariances,
                self.old_conics, torch.sigmoid(self.old_opacities), samples)
            translation_sample = self.sampler.sample_gaussians() # n, c

            bc_mask = self.bc_mask(samples)
            pde_loss += torch.mean(bc_mask * (translation_sample - self.u_samples[-2][...,:2]) ** 2)
            pde_loss += torch.mean(bc_mask * (ux[:,0,0] + ux[:,1,1]) ** 2)
            pde_loss += torch.mean(bc_mask * (ut[...,:2] - rhs) ** 2)


        bc_loss += torch.mean(bc_u_sample ** 2)
        conservation_loss += torch.mean(self.translation ** 2)
        conservation_loss += torch.mean(self.dtransform ** 2)
        conservation_loss += torch.mean(self.dscale ** 2)
        if self.train_opacity:
            conservation_loss += torch.mean(self.dopacities ** 2)

        return self.pde_weight * pde_loss,\
               self.bc_weight * bc_loss,\
               self.conservation_weight * conservation_loss

    def generate_images(self, res):
        initial_opacities = torch.sigmoid(self.initial_opacities)
        opacities = torch.sigmoid(self.opacities)
        if self.problem == Problem.WAVE:
            img1 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics, initial_opacities,
                self.initial_u[...,0], res, res, self.scale
            ).detach().cpu().numpy()

            img2 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics, initial_opacities,
                self.initial_u[...,1], res, res, self.scale
            ).detach().cpu().numpy()

            img3 = gaussians.sample_gaussians_img(
                self.means, self.conics, opacities, self.u[...,0], res, res, self.scale
            ).detach().cpu().numpy()

            img4 = gaussians.sample_gaussians_img(
                self.means, self.conics, opacities, self.u[...,1], res, res, self.scale
            ).detach().cpu().numpy()
        elif self.problem == Problem.NAVIER_STOKES:
            img1 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics, initial_opacities,
                torch.sqrt((self.initial_u[...,:2] ** 2).sum(-1)), res, res, self.scale
            ).detach().cpu().numpy()

            img2 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics, initial_opacities,
                self.initial_u[...,-1], res, res, self.scale
            ).detach().cpu().numpy()

            mag_dir = torch.stack((
                self.u[...,0], self.u[...,1], torch.sqrt((self.u[...,:2] ** 2).sum(-1))), dim=-1)
            img3 = gaussians.sample_gaussians_img(
                self.means, self.conics, opacities,
                mag_dir, res, res, self.scale
            ).detach().cpu().numpy()

            img4 = gaussians.sample_gaussians_img(
                self.means, self.conics, opacities, self.u[...,-1], res, res, self.scale
            ).detach().cpu().numpy()
        else:
            img1 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics, initial_opacities,
                self.initial_u, res, res, self.scale
            ).detach().cpu().numpy()

            img2 = gaussians.sample_gaussians_img(
                self.means, self.conics, opacities, self.u, res, res, self.scale
            ).detach().cpu().numpy()

            img3 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics, opacities, self.u, res, res, self.scale
            ).detach().cpu().numpy()

            img4 = gaussians.sample_gaussians_img(
                self.means, self.conics, initial_opacities, self.initial_u, res, res, self.scale
            ).detach().cpu().numpy()

        return img1, img2, img3, img4

    def plot_gaussians(self):
        if self.train_opacity:
            return gaussians.plot_gaussians(
                self.means, self.covariances, torch.sigmoid(self.opacities), self.u, self.scale)
        else:
            return gaussians.plot_gaussians(
                self.means, self.covariances, torch.sigmoid(self.opacities) * 0.25, self.u, self.scale)
