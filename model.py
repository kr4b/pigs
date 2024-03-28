import time
import enum

import numpy as np
import torch

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
            nn.Linear(d + d + d * (d - 1) // 2 + c, 30),
            # nn.LayerNorm(64),
            activation,
        )
        self.linear = nn.Sequential(
            nn.Linear(80, 80),
            # nn.LayerNorm(256),
            activation,
            nn.Linear(80, 80),
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
    def __init__(self, problem, nx, ny, d, dx, kernel_size, rule, scale):
        super(Model, self).__init__()
        self.problem = problem
        self.nx = nx
        self.ny = ny
        self.d = d
        self.dx = dx
        self.kernel_size = kernel_size
        self.rule = rule
        self.scale = scale

        self.nu = 1.0 / (100.0 * np.pi)
        self.inv_rho = 0.1
        self.mu = 0.1

        self.pde_weight = 1.0
        self.bc_weight = 1.0
        self.conservation_weight = 0.1

        self.channels = 1

        tx = torch.linspace(-1, 1, nx).cuda() * scale
        ty = torch.linspace(-1, 1, ny).cuda() * scale
        gx, gy = torch.meshgrid((tx,ty), indexing="ij")
        self.initial_means = torch.stack((gx,gy), dim=-1)
        self.initial_means = self.initial_means.reshape(-1, d)
        scaling = torch.ones((nx*ny,d), device="cuda") * -4.0
        # scaling[10*ny+20] = -3.5
        # scaling[8*ny+20] = -3.5
        # scaling[20*ny+20] = -3.5
        # scaling[22*ny+20] = -3.5

        self.initial_scaling = torch.exp(scaling) * scale
        self.transform_size = d * (d - 1) // 2
        self.initial_transform = torch.zeros((nx*ny,self.transform_size), device="cuda")

        if problem == Problem.POISSON:
            self.initial_u = torch.zeros((nx*ny), device="cuda")
        elif problem == Problem.WAVE:
            self.channels = 2
            # self.initial_means = self.initial_means * 0.05
            # self.initial_u = torch.cat((
            #     torch.zeros((nx*ny, 1), device="cuda"),
            #     torch.ones((nx*ny, 1), device="cuda")
            # ), dim=-1)
            self.initial_u = torch.zeros((nx*ny, 2), device="cuda")
            self.initial_u[(ny//2-1) * nx + nx//2-1] = 0.5
            self.initial_u[ny//2 * nx + nx//2-1] = 0.5
            self.initial_u[(ny//2-1) * nx + nx//2] = 0.5
            self.initial_u[ny//2 * nx + nx//2] = 0.5

        elif problem == Problem.NAVIER_STOKES:
            self.channels = 3
            self.initial_u = torch.cat((
                torch.ones((ny, 1), device="cuda") * 0.25,
                torch.zeros((ny, 2), device="cuda"),
            ), dim=-1)
            self.initial_scaling = self.initial_scaling[ny:2*ny]
            self.initial_means = self.initial_means[ny:2*ny]
            self.initial_transform = self.initial_transform[ny:2*ny]
            # self.initial_u[2*ny+2:2*ny+ny-3,0] = 1.0
            # self.initial_u[3*ny+2:3*ny+ny-3,0] = 1.0
        else:
            sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, d, 1)
            samples = self.initial_means.unsqueeze(-1) - sample_mean
            conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1 * self.scale)
            powers = -0.5 * (samples.transpose(-1, -2) @ (conics @ samples))
            self.initial_u = torch.exp(powers).squeeze(-1) / 5.0

        keep_mask = torch.norm(self.initial_u, dim=-1) > 0.001
        self.initial_means = self.initial_means[keep_mask]
        self.initial_u = self.initial_u[keep_mask]
        self.initial_scaling = self.initial_scaling[keep_mask]
        self.initial_transform = self.initial_transform[keep_mask]

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

        in_channels = self.channels * 2 + 1

        if self.problem == Problem.NAVIER_STOKES:
            self.solution_model_v = Network(
                in_channels, 2, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
            self.solution_model_p = Network(
                in_channels, 1, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
            self.optimization_model_v = Network(
                in_channels, self.channels, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
            self.optimization_model_p = Network(
                in_channels, self.channels, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
        else:
            self.solution_model = Network(
                in_channels, self.channels, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
            self.optimization_model = Network(
                in_channels, self.channels, self.channels, d, self.kernel_size, nn.Tanh()).cuda()

        self.translation_model = Network(
            in_channels, d, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
        self.scale_model = Network(
            in_channels, d, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
        self.transform_model = Network(
            in_channels, self.transform_size, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
        self.translation_optimization_model = Network(
            in_channels, d, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
        self.scale_optimization_model = Network(
            in_channels, d, self.channels, d, self.kernel_size, nn.Tanh()).cuda()
        self.transform_optimization_model = Network(
            in_channels, self.transform_size, self.channels, d, self.kernel_size, nn.Tanh()).cuda()

        self.sampler = GaussianSampler(False)
        self.sample_kernel = \
            gaussians.region_kernel(self.kernel_size, self.dx, self.d).reshape(1, -1, d)

        self.reset()

    def reset(self):
        self.u = self.initial_u
        self.means = self.initial_means
        self.scaling = self.initial_scaling
        self.transform = self.initial_transform
        self.covariances = self.initial_covariances
        self.conics = self.initial_conics

        self.clear()

    def clear(self):
        # self.ut_samples = []
        self.ux_samples = []
        self.uxx_samples = []
        self.u_samples = []
        self.bc_u_samples = []

    def parameters_solve(self):
        parameters = []

        if self.problem == Problem.NAVIER_STOKES:
            parameters.extend(self.solution_model_v.parameters())
            parameters.extend(self.solution_model_p.parameters())
        else:
            parameters.extend(self.solution_model.parameters())

        parameters.extend(self.translation_model.parameters())
        parameters.extend(self.scale_model.parameters())
        parameters.extend(self.transform_model.parameters())

        return parameters

    def parameters_optim(self):
        parameters = []

        if self.problem == Problem.NAVIER_STOKES:
            parameters.extend(self.optimization_model_v.parameters())
            parameters.extend(self.optimization_model_p.parameters())
        else:
            parameters.extend(self.optimization_model.parameters())

        parameters.extend(self.translation_optimization_model.parameters())
        parameters.extend(self.scale_optimization_model.parameters())
        parameters.extend(self.transform_optimization_model.parameters())

        return parameters

    def detach(self):
        self.u = self.u.detach()
        self.means = self.means.detach()
        self.scaling = self.scaling.detach()
        self.transform = self.transform.detach()
        self.covariances = self.covariances.detach()
        self.conics = self.conics.detach()
        self.translation = self.translation.detach()
        self.dscale = self.dscale.detach()
        self.dtransform = self.dtransform.detach()

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
            return torch.stack((
                u[...,1],
                10 * (uxx[...,0,0,0] + uxx[...,1,1,0]) - 0.1 * u[...,1],
            ), dim=-1)

        elif self.problem == Problem.NAVIER_STOKES:
            return self.mu * (uxx[:,0,0,:2] + uxx[:,1,1,:2]) \
                 - (u[:,:2].reshape(-1, 1, 2) * ux[...,:2]).sum(-1) \
                 - self.inv_rho * ux[...,-1]

        else:
            raise ValueError("Unexpected PDE problem:", self.problem)

    def forward(self, dt):
        self.prev_means = self.means
        self.prev_u = self.u
        self.prev_covariances = self.covariances
        self.prev_conics = self.conics

        with torch.no_grad():
            samples = (
                self.means.reshape(self.means.shape[0], 1, self.d) + self.sample_kernel
            ).reshape(-1, self.d) # n*k*k, d
            bc_mask = self.bc_mask(samples) # n*k*k, 1
            self.sampler.preprocess(self.means, self.u, self.covariances, self.conics, samples)
            u = self.sampler.sample_gaussians() # n*k*k, c
            ux = self.sampler.sample_gaussians_derivative() # n*k*k, d, c
            uxx = self.sampler.sample_gaussians_laplacian() # n*k*k, d, d, c
            pde = dt * self.pde_rhs(samples, u, ux, uxx).reshape(u.shape[0], -1) # n*k*k, -1
            in_samples = torch.cat((u, pde, bc_mask), -1)

        in_samples = in_samples.reshape(
            self.means.shape[0], self.kernel_size * self.kernel_size, -1
        ).transpose(-1, -2).reshape(
            self.means.shape[0], -1, self.kernel_size, self.kernel_size
        ) # n, -1, k, k

        in_params = torch.cat((
            self.means.reshape(-1, 2),
            self.u.reshape(-1, self.channels),
            self.scaling.reshape(-1, 2),
            self.transform.reshape(-1, self.transform_size),
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

        covariances = gaussians.build_covariances(scaling, transform)

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
            self.covariances = covariances
            self.conics = torch.inverse(self.covariances)

    def optimize(self, dt, threshold):
        keep_mask = torch.logical_and(
            torch.norm(self.u, dim=-1) > 0.01,
            torch.sum(self.scaling, dim=-1) < 1.0,
        )

        kk = self.kernel_size * self.kernel_size
        samples = (
            self.means[keep_mask].reshape(self.means[keep_mask].shape[0], 1, self.d) \
            + self.sample_kernel
        ).reshape(-1, self.d) # n*k*k, d

        self.sampler.preprocess(self.means[keep_mask], self.u[keep_mask], self.covariances[keep_mask], self.conics[keep_mask], samples)
        u = self.sampler.sample_gaussians() # n*k*k, c
        ux = self.sampler.sample_gaussians_derivative() # n*k*k, d, c
        uxx = self.sampler.sample_gaussians_laplacian() # n*k*k, d, d, c

        self.sampler.preprocess(self.prev_means, self.prev_u, self.prev_covariances, self.prev_conics, samples)
        prev_u = self.sampler.sample_gaussians() # n*k*k, c
        prev_ux = self.sampler.sample_gaussians_derivative() # n*k*k, d, c
        prev_uxx = self.sampler.sample_gaussians_laplacian() # n*k*k, d, d, c

        ut = u.reshape(u.shape[0] // kk, kk, -1) - prev_u.reshape(u.shape[0] // kk, kk, -1)

        time_samples = torch.rand(samples.shape[0], device="cuda")
        u = time_samples.reshape(-1, 1) * u + (1 - time_samples.reshape(-1, 1)) * prev_u
        ux = time_samples.reshape(-1, 1, 1) * ux + (1 - time_samples.reshape(-1, 1, 1)) * prev_ux
        uxx = time_samples.reshape(-1, 1, 1, 1) * uxx \
            + (1 - time_samples.reshape(-1, 1, 1, 1)) * prev_uxx

        rhs = dt * self.pde_rhs(samples, u, ux, uxx).reshape(u.shape[0]//kk, kk, -1) # n,k*k,-1

        pde_loss = torch.zeros(u.shape[0] // kk, device="cuda")

        if self.problem == Problem.DIFFUSION:
            pde_loss += torch.mean((ut - rhs) ** 2, (-1, -2))

        elif self.problem == Problem.BURGERS:
            pde_loss += torch.mean((ut - rhs) ** 2, (-1, -2))

        elif self.problem == Problem.POISSON:
            pde_loss += torch.mean(rhs ** 2, (-1, -2))

        elif self.problem == Problem.WAVE:
            pde_loss += 0.01 * torch.mean((ut[...,0] - rhs[...,0]) ** 2, (-1, -2))
            pde_loss += torch.mean((ut[...,1] - rhs[...,1]) ** 2, (-1, -2))

        split_indices = pde_loss > threshold

        eigvals, eigvecs = torch.linalg.eig(self.covariances[keep_mask][split_indices])
        eigval_max, indices = torch.max(eigvals.real.abs(), dim=-1)
        eigvec_max = torch.gather(
            eigvecs.real, -1, indices.reshape(-1, 1, 1).expand(eigvals.shape[0], self.d, 1))
        pc = eigval_max.reshape(-1, 1) * eigvec_max.squeeze(-1)

        print(torch.sum(keep_mask).item(), torch.sum(split_indices).item())

        self.means[keep_mask][split_indices] = self.means[keep_mask][split_indices] - pc
        self.means = torch.cat((
            self.means[keep_mask], self.means[keep_mask][split_indices] + 2 * pc), 0)
        self.u[keep_mask][split_indices] /= 2
        self.u = torch.cat((self.u[keep_mask], self.u[keep_mask][split_indices]), 0)
        self.scaling = torch.cat((
            self.scaling[keep_mask], self.scaling[keep_mask][split_indices]), 0)
        self.transform = torch.cat((
            self.transform[keep_mask], self.transform[keep_mask][split_indices]), 0)
        self.covariances = gaussians.build_covariances(self.scaling, self.transform)
        self.conics = torch.inverse(self.covariances)

        samples = (
            self.means.reshape(self.means.shape[0], 1, self.d) + self.sample_kernel
        ).reshape(-1, self.d) # n*k*k, d
        bc_mask = self.bc_mask(samples) # n*k*k, 1

        self.sampler.preprocess(self.means, self.u, self.covariances, self.conics, samples)
        u = self.sampler.sample_gaussians() # n*k*k, c
        ux = self.sampler.sample_gaussians_derivative() # n*k*k, d, c
        uxx = self.sampler.sample_gaussians_laplacian() # n*k*k, d, d, c

        self.sampler.preprocess(self.prev_means, self.prev_u, self.prev_covariances, self.prev_conics, samples)
        prev_u = self.sampler.sample_gaussians() # n*k*k, c
        prev_ux = self.sampler.sample_gaussians_derivative() # n*k*k, d, c
        prev_uxx = self.sampler.sample_gaussians_laplacian() # n*k*k, d, d, c

        rhs = dt * self.pde_rhs(samples, u, ux, uxx) # n,k*k,-1
        ut = u - prev_u # n, k*k, -1

        if self.problem == Problem.DIFFUSION or self.problem == Problem.BURGERS or self.problem == Problem.WAVE:
            residual = ut - rhs

        elif self.problem == Problem.POISSON:
            residual = rhs

        in_samples = torch.cat((u, residual, bc_mask), -1)

        in_samples = in_samples.reshape(
            self.means.shape[0], self.kernel_size * self.kernel_size, -1
        ).transpose(-1, -2).reshape(
            self.means.shape[0], -1, self.kernel_size, self.kernel_size
        ) # n, -1, k, k

        in_params = torch.cat((
            self.means.reshape(-1, 2),
            self.u.reshape(-1, self.channels),
            self.scaling.reshape(-1, 2),
            self.transform.reshape(-1, self.transform_size),
        ), dim=-1)

        if self.problem == Problem.NAVIER_STOKES:
            delta_v = self.optimization_model_v(
                in_samples, in_params).reshape(self.means.shape[0], 2)
            delta_p = self.optimization_model_p(
                in_samples, in_params).reshape(self.means.shape[0], 1)

            deltas = torch.cat((delta_v, delta_p), dim=-1)
        else:
            deltas = self.optimization_model(
                in_samples, in_params).reshape(self.means.shape[0], self.channels)

        self.u = self.u + deltas

        self.translation = self.translation_optimization_model(
            in_samples, in_params).reshape(self.means.shape[0], -1)
        self.means = self.means + self.translation

        self.dscale = self.scale_optimization_model(
            in_samples, in_params).reshape(self.means.shape[0], -1)
        scale = torch.exp(self.dscale)
        self.scaling = self.scaling * scale

        self.dtransform = self.transform_optimization_model(
            in_samples, in_params).reshape(self.means.shape[0], -1)
        self.transform = self.transform + self.dtransform

        self.covariances = gaussians.build_covariances(self.scaling, self.transform)
        self.conics = torch.inverse(self.covariances)

    def sample(self, samples, bc_samples):
        if len(self.u_samples) > 1:
            self.u_samples.pop()
            self.ux_samples.pop()
            self.uxx_samples.pop()
            self.bc_u_samples.pop()

        self.sampler.preprocess(self.means, self.u, self.covariances, self.conics, samples)
        u_sample = self.sampler.sample_gaussians() # n, c
        ux = self.sampler.sample_gaussians_derivative() # n, d, c
        uxx = self.sampler.sample_gaussians_laplacian() # n, d, d, c

        self.sampler.preprocess(self.means, self.u, self.covariances, self.conics, bc_samples)
        bc_u_sample = self.sampler.sample_gaussians() # n, c

        self.u_samples.append(u_sample)
        self.bc_u_samples.append(bc_u_sample)
        self.ux_samples.append(ux)
        self.uxx_samples.append(uxx)

    def compute_loss(self, dt, samples, time_samples, bc_samples):
        self.sample(samples, bc_samples)

        if self.rule == IntegrationRule.TRAPEZOID:
            u_sample = time_samples.reshape(-1, 1) * self.u_samples[-1] \
                     + (1 - time_samples.reshape(-1, 1)) * self.u_samples[-2]
            ux = time_samples.reshape(-1, 1, 1) * self.ux_samples[-1] \
               + (1 - time_samples.reshape(-1, 1, 1)) * self.ux_samples[-2]
            uxx = time_samples.reshape(-1, 1, 1, 1) * self.uxx_samples[-1] \
                + (1 - time_samples.reshape(-1, 1, 1, 1)) * self.uxx_samples[-2]
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

        ut = self.u_samples[-1] - self.u_samples[-2]
        bc_u_sample = self.bc_u_samples[-1]

        pde_loss = torch.zeros(1, device="cuda")
        bc_loss = torch.zeros(1, device="cuda")
        conservation_loss = torch.zeros(1, device="cuda")

        rhs = dt * self.pde_rhs(samples, u_sample, ux, uxx)

        if self.problem == Problem.DIFFUSION:
            pde_loss += torch.mean((ut - rhs) ** 2)

        elif self.problem == Problem.BURGERS:
            pde_loss += torch.mean((ut - rhs) ** 2)

        elif self.problem == Problem.POISSON:
            pde_loss += torch.mean(rhs ** 2)

        elif self.problem == Problem.WAVE:
            pde_loss += 0.01 * torch.mean((ut[...,0] - rhs[...,0]) ** 2)
            pde_loss += torch.mean((ut[...,1] - rhs[...,1]) ** 2)

        elif self.problem == Problem.NAVIER_STOKES:
            self.sampler.preprocess(
                self.prev_means, self.translation, self.prev_covariances, self.prev_conics, samples)
            translation_sample = self.sampler.sample_gaussians() # n, c

            bc_mask = self.bc_mask(samples)
            pde_loss += torch.mean(bc_mask * (translation_sample - self.u_samples[-2][...,:2]) ** 2)
            pde_loss += torch.mean(bc_mask * (ux[:,0,0] + ux[:,1,1]) ** 2)
            pde_loss += torch.mean(bc_mask * (ut[...,:2] - rhs) ** 2)

        bc_loss += torch.mean(bc_u_sample ** 2)
        conservation_loss += torch.mean(self.translation ** 2)
        conservation_loss += torch.mean(self.dtransform ** 2)
        conservation_loss += torch.mean(self.dscale ** 2)
        conservation_loss += torch.mean(self.scaling ** 2)

        return self.pde_weight * pde_loss,\
               self.bc_weight * bc_loss,\
               self.conservation_weight * conservation_loss

    def generate_images(self, res):
        if self.problem == Problem.WAVE:
            img1 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics, self.initial_u[...,0], res, res, self.scale
            ).detach().cpu().numpy()

            img2 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics, self.initial_u[...,1], res, res, self.scale
            ).detach().cpu().numpy()

            img3 = gaussians.sample_gaussians_img(
                self.means, self.conics, self.u[...,0], res, res, self.scale
            ).detach().cpu().numpy()

            img4 = gaussians.sample_gaussians_img(
                self.means, self.conics, self.u[...,1], res, res, self.scale
            ).detach().cpu().numpy()
        elif self.problem == Problem.NAVIER_STOKES:
            img1 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics,
                torch.sqrt((self.initial_u[...,:2] ** 2).sum(-1)), res, res, self.scale
            ).detach().cpu().numpy()

            img2 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics, self.initial_u[...,-1], res, res, self.scale
            ).detach().cpu().numpy()

            mag_dir = torch.stack((
                self.u[...,0], self.u[...,1], torch.sqrt((self.u[...,:2] ** 2).sum(-1))), dim=-1)
            img3 = gaussians.sample_gaussians_img(
                self.means, self.conics, mag_dir, res, res, self.scale
            ).detach().cpu().numpy()

            img4 = gaussians.sample_gaussians_img(
                self.means, self.conics, self.u[...,-1], res, res, self.scale
            ).detach().cpu().numpy()
        else:
            img1 = gaussians.sample_gaussians_img(
                self.initial_means, self.initial_conics, self.initial_u, res, res, self.scale
            ).detach().cpu().numpy()

            img2 = gaussians.sample_gaussians_img(
                self.means, self.conics, self.u, res, res, self.scale
            ).detach().cpu().numpy()

            img3 = gaussians.sample_gaussians_img(
            #    self.initial_means, self.initial_conics, self.u, res, res, self.scale
                self.means, self.conics, self.u, res, res, self.scale
            ).detach().cpu().numpy()

            img4 = gaussians.sample_gaussians_img(
            #   self.means, self.conics, self.initial_u, res, res, self.scale
                self.means, self.conics, self.u, res, res, self.scale
            ).detach().cpu().numpy()

        return img1, img2, img3, img4

    def plot_gaussians(self):
        return gaussians.plot_gaussians(self.means, self.covariances, self.u, self.scale)
