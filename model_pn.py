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
    TEST = enum.auto()

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

class RBFAct(nn.Module):
    def __init__(self, in_dim):
        super(RBFAct, self).__init__() 
        self.b = nn.Parameter(torch.ones(1), requires_grad=True)
        self.c = nn.Parameter(torch.zeros(in_dim), requires_grad=True)

    def forward(self, x):
        return torch.exp(-self.b * (x - self.c) ** 2)

LATENT_SIZE = 16
L1_SIZE = 16
L2_SIZE = 32
L3_SIZE = 48
EMBEDDING_SIZE = 25
ATTENTION_HEADS = 2

class LatentTransform(nn.Module):
    def __init__(self, in_channels, activation):
        super(LatentTransform, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, L1_SIZE, 1),
            # nn.GroupNorm(1, L1_SIZE),
            activation((L1_SIZE, 1)),
            nn.Conv1d(L1_SIZE, L2_SIZE, 1),
            # nn.GroupNorm(1, L2_SIZE),
            activation((L2_SIZE, 1)),
            nn.Conv1d(L2_SIZE, LATENT_SIZE, 1),
            # nn.GroupNorm(1, LATENT_SIZE),
            activation((LATENT_SIZE, 1)),
        )

    def forward(self, x):
        return self.layers(x)

class TransformNet(nn.Module):
    def __init__(self, d, activation):
        super(TransformNet, self).__init__()
        self.d = d
        self.layers = nn.Sequential(
            nn.Linear(LATENT_SIZE, L3_SIZE),
            # nn.LayerNorm(L3_SIZE),
            activation(L3_SIZE),
            nn.Linear(L3_SIZE, L2_SIZE),
            # nn.LayerNorm(L2_SIZE),
            activation(L2_SIZE),
            nn.Linear(L2_SIZE, d*d),
        )

    def forward(self, x):
        return torch.eye(self.d, device="cuda").unsqueeze(0) \
             + self.layers(x).reshape(-1, self.d, self.d) # 1, d, d

class InputTransform(nn.Module):
    def __init__(self, in_dims, c, d, pde_size, activation):
        super(InputTransform, self).__init__()
        self.c = c
        self.d = d
        self.pde_size = pde_size

        self.latent_net = LatentTransform(in_dims + d, activation)
        self.transform_net = TransformNet(d, activation)
        self.transform_u_net = TransformNet(c, activation)
        self.transform_ux_net = TransformNet(d*c, activation)
        self.transform_uxx_net = TransformNet(d*c, activation)
        self.transform_pde_net = TransformNet(pde_size, activation)

    def forward(self, means, full_covariances, u, boundaries,
                sample_u, sample_ux, sample_uxx, sample_pde):
        means = means.unsqueeze(0) # 1, n, d
        covariances = full_covariances.reshape(1, -1, self.d*self.d) # 1, n, d*d
        u = u.unsqueeze(0) # 1, n, c
        sample_u = sample_u.unsqueeze(0) # 1, n, c
        ux = sample_ux.unsqueeze(0) # 1, n, d*c
        uxx = sample_uxx.unsqueeze(0) # 1, n, d*c
        pde = sample_pde.unsqueeze(0) # 1, n, pde_size
        params = torch.cat(
            (means, covariances, u, boundaries, sample_u, ux, uxx, pde), dim=-1).transpose(1, 2)

        latent = self.latent_net(params).mean(-1) # 1, LATENT_SIZE
        self.transform = self.transform_net(latent) # 1, d, d
        self.transform_u = self.transform_u_net(latent) # 1, c, c
        self.transform_ux = self.transform_ux_net(latent) # 1, d*c, d*c
        self.transform_uxx = self.transform_uxx_net(latent) # 1, d*c, d*c
        self.transform_pde = self.transform_pde_net(latent) # 1, pde_size, pde_size

        covariances = full_covariances.reshape(-1, self.d, self.d)
        means = means.reshape(-1, self.d, 1)
        u = u.reshape(-1, self.c, 1)
        sample_u = sample_u.reshape(-1, self.c, 1)
        ux = ux.reshape(-1, self.d*self.c, 1)
        uxx = uxx.reshape(-1, self.d*self.c, 1)
        pde = pde.reshape(-1, self.pde_size, 1)

        return self.transform @ means, \
               self.transform @ covariances, \
               self.transform_u @ u, \
               self.transform_u @ sample_u, \
               self.transform_ux @ ux, \
               self.transform_uxx @ uxx, \
               self.transform_pde @ pde \

    def transform_gaussians(self, means, full_covariances, u, sample_u, ux, uxx, pde):
        covariances = full_covariances.reshape(-1, self.d, self.d)
        means = means.reshape(-1, self.d, 1)
        u = u.reshape(-1, self.c, 1)
        sample_u = sample_u.reshape(-1, self.c, 1)
        ux = ux.reshape(-1, self.d*self.c, 1)
        uxx = uxx.reshape(-1, self.d*self.c, 1)
        pde = pde.reshape(-1, self.pde_size, 1)

        return self.transform @ means, \
               self.transform @ covariances, \
               self.transform_u @ u, \
               self.transform_u @ sample_u, \
               self.transform_ux @ ux, \
               self.transform_uxx @ uxx, \
               self.transform_pde @ pde \

def delta_network(m, in_dims, c, d, activation):
    transform_size = d * (d-1) // 2
    l = ATTENTION_HEADS//2 + 1
    return nn.Sequential(
        nn.Linear((ATTENTION_HEADS + 1) * LATENT_SIZE, l*LATENT_SIZE),
        # nn.LayerNorm(l*LATENT_SIZE),
        activation(l*LATENT_SIZE),
        nn.Linear(l*LATENT_SIZE, LATENT_SIZE),
        # nn.LayerNorm(LATENT_SIZE),
        activation(LATENT_SIZE),
        nn.Linear(LATENT_SIZE, LATENT_SIZE),
        # nn.LayerNorm(LATENT_SIZE),
        activation(LATENT_SIZE),
        nn.Linear(LATENT_SIZE, L3_SIZE),
        # nn.LayerNorm(L3_SIZE),
        activation(L3_SIZE),
        nn.Linear(L3_SIZE, L2_SIZE),
        # nn.LayerNorm(L2_SIZE),
        activation(L2_SIZE),
        nn.Linear(L2_SIZE, m * (d + d + transform_size + c)),
    )

class DynamicsNetwork(nn.Module):
    def __init__(self, c, d, pde_size, split_size, activation):
        super(DynamicsNetwork, self).__init__()
        self.c = c
        self.d = d
        self.transform_size = d * (d-1) // 2
        self.pde_size = pde_size
        self.split_size = split_size
        self.in_dims = 1 + d * d + 2*c + 2*d*c + pde_size

        self.input_transform = InputTransform(self.in_dims, c, d, pde_size, activation)
        self.input_projection = nn.Sequential(
            nn.Linear(self.in_dims, L1_SIZE),
            # nn.LayerNorm(L1_SIZE),
            activation(L1_SIZE),
            nn.Linear(L1_SIZE, L2_SIZE),
            # nn.LayerNorm(L2_SIZE),
            activation(L2_SIZE),
            nn.Linear(L2_SIZE, L3_SIZE),
            # nn.LayerNorm(L3_SIZE),
            activation(L3_SIZE),
            nn.Linear(L3_SIZE, LATENT_SIZE),
        )
        self.distance_transform = nn.Parameter(
            torch.rand((ATTENTION_HEADS, LATENT_SIZE, EMBEDDING_SIZE*2), device="cuda") * 2.0 - 1.0)
        self.transform = nn.Parameter(
            torch.rand((ATTENTION_HEADS, LATENT_SIZE, LATENT_SIZE), device="cuda") * 2.0 - 1.0)
        self.query_transform = nn.ParameterList([nn.Sequential(
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            # nn.LayerNorm(LATENT_SIZE),
            activation(LATENT_SIZE),
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            # nn.LayerNorm(LATENT_SIZE),
            activation(LATENT_SIZE),
            nn.Linear(LATENT_SIZE, (LATENT_SIZE+L1_SIZE)//2),
            # nn.LayerNorm((LATENT_SIZE+L1_SIZE)//2),
            activation((LATENT_SIZE+L1_SIZE)//2),
            nn.Linear((LATENT_SIZE+L1_SIZE)//2, L1_SIZE),
        ) for _ in range(ATTENTION_HEADS)])
        self.key_transform = nn.ParameterList([nn.Sequential(
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            # nn.LayerNorm(LATENT_SIZE),
            activation(LATENT_SIZE),
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            # nn.LayerNorm(LATENT_SIZE),
            activation(LATENT_SIZE),
            nn.Linear(LATENT_SIZE, (LATENT_SIZE+L1_SIZE)//2),
            # nn.LayerNorm((LATENT_SIZE+L1_SIZE)//2),
            activation((LATENT_SIZE+L1_SIZE)//2),
            nn.Linear((LATENT_SIZE+L1_SIZE)//2, L1_SIZE),
        ) for _ in range(ATTENTION_HEADS)])
        self.frequencies = nn.Parameter(
            torch.randn((EMBEDDING_SIZE-1) // d // 2, device="cuda") * 10,
            requires_grad=False
        )
        self.delta_net = delta_network(1, self.in_dims, c, d, activation)

    def forward(self, means, full_covariances, u, boundaries,
                sample_u, sample_ux, sample_uxx, sample_pde):
        boundaries = boundaries.reshape(1, -1, 1)

        t_means, t_covariances, t_u, t_sample_u, t_ux, t_uxx, t_pde = \
            self.input_transform(means, full_covariances, u, boundaries,
                                 sample_u, sample_ux, sample_uxx, sample_pde)

        t_means = t_means.reshape(1, -1, self.d)
        t_covariances = t_covariances.reshape(1, -1, self.d*self.d)
        t_u = t_u.reshape(1, -1, self.c)
        t_sample_u = t_sample_u.reshape(1, -1, self.c)
        t_ux = t_ux.reshape(1, -1, self.d*self.c)
        t_uxx = t_uxx.reshape(1, -1, self.d*self.c)
        t_pde = t_pde.reshape(1, -1, self.pde_size)
        self.t_params = torch.cat(
            (t_covariances, t_u, boundaries, t_sample_u, t_ux, t_uxx, t_pde), dim=-1)

        self.global_features = self.input_projection(self.t_params) # 1, n, LATENT_SIZE

    def _compute_deltas(self, sampler, t_params, features, delta_net, size):
        b, n, l = features.shape

        self.local_global_features = features
        sampler.preprocess_aggregate()

        for i in range(ATTENTION_HEADS):
            queries = self.query_transform[i](features).reshape(b, n, -1) # 1, n, L1_SIZE
            keys = self.key_transform[i](features).reshape(b, n, -1) # 1, n, L1_SIZE
            neighbor_features = sampler.aggregate_neighbors(
                features.squeeze(0), self.transform[i], queries.squeeze(0),
                keys.squeeze(0), self.frequencies, self.distance_transform[i])
            self.local_global_features = torch.cat(
                (self.local_global_features, neighbor_features.unsqueeze(0)), dim=-1)

        deltas = delta_net(self.local_global_features)
        dmeans = deltas[...,:self.d*size].squeeze(0)
        dscaling = deltas[...,self.d*size:2*self.d*size].squeeze(0)
        dtransforms = deltas[...,2*self.d*size:(2*self.d+self.transform_size)*size].squeeze(0)
        du = deltas[...,-self.c*size:].squeeze(0)

        return dmeans, dscaling, dtransforms, du

    def compute_deltas(self, sampler):
        features = self.global_features # 1, n, LATENT_SIZE
        return self._compute_deltas(sampler, self.t_params, features, self.delta_net, 1)

    def transform_gaussians(self, means, full_covariances, u,
                            sample_u, sample_ux, sample_uxx, sample_pde):
        boundaries = torch.zeros((1, means.shape[0], 1), dtype=torch.bool, device="cuda")

        t_means, t_covariances, t_u, t_sample_u, t_ux, t_uxx, t_pde = \
            self.input_transform.transform_gaussians(
                means, full_covariances, u, sample_u, sample_ux, sample_uxx, sample_pde)

        t_means = t_means.reshape(1, -1, self.d)
        t_covariances = t_covariances.reshape(1, -1, self.d*self.d)
        t_u = t_u.reshape(1, -1, self.c)
        t_sample_u = t_sample_u.reshape(1, -1, self.c)
        t_ux = t_ux.reshape(1, -1, self.d*self.c)
        t_uxx = t_uxx.reshape(1, -1, self.d*self.c)
        t_pde = t_pde.reshape(1, -1, self.pde_size)
        t_params = torch.cat((t_covariances, t_u, boundaries, t_sample_u, t_ux, t_uxx, t_pde),dim=-1)
        self.t_params = torch.cat((self.t_params, t_params), dim=1)

        global_features = self.input_projection(t_params) # 1, n, L1_SIZE
        self.global_features = torch.cat((self.global_features, global_features), dim=1)


class Model(nn.Module):
    def __init__(self, problem, rule, nx, ny, d, scale):
        super(Model, self).__init__()
        self.problem = problem
        self.rule = rule
        self.nx = nx
        self.ny = ny
        self.d = d
        self.scale = scale

        if problem == Problem.TEST:
            self.pde_weight = 10.0
            self.bc_weight = 2.0
            self.conservation_weight = 0.5
            self.initial_weight = 1.0
            self.du_weight = 4.0
            self.dmean_weight = 4.0
            self.dtransform_weight = 1.0
            self.dscale_weight = 1.0
        else:
            self.pde_weight = 1.0
            self.bc_weight = 1.0
            self.conservation_weight = 0.1
            self.initial_weight = 2.0
            self.du_weight = 1.0
            self.dmean_weight = 2.0
            self.dtransform_weight = 2.0
            self.dscale_weight = 2.0

        self.split_size = 2

        if problem == Problem.BURGERS:
            self.nu = 1.0 / (10.0 * np.pi)
        elif problem == Problem.NAVIER_STOKES:
            self.nu = 1e-3

        tx = torch.linspace(-1, 1, nx).cuda() * scale
        ty = torch.linspace(-1, 1, ny).cuda() * scale
        gx, gy = torch.meshgrid((tx,ty), indexing="ij")
        self.initial_means = torch.stack((gx,gy), dim=-1)# * 0
        self.initial_means = self.initial_means.reshape(-1, d)

        scaling = torch.ones((nx*ny,d), device="cuda") * -4.0
        self.initial_scaling = torch.exp(scaling) * scale

        self.transform_size = d * (d - 1) // 2
        self.initial_transforms = torch.zeros((nx*ny,self.transform_size), device="cuda")

        if problem == Problem.BURGERS or problem == Problem.DIFFUSION:
            self.channels = 1

            sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, d, 1)
            samples = self.initial_means.unsqueeze(-1) - sample_mean
            conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1 * scale)
            powers = -0.5 * (samples.transpose(-1, -2) @ (conics @ samples))
            self.initial_u = torch.exp(powers).squeeze(-1) / 3.0

        elif problem == Problem.WAVE:
            self.channels = 2
            self.initial_u = torch.zeros((nx*ny, self.channels), device="cuda")
            for i in range(-2, 3):
                for j in range(-2, 3):
                    self.initial_u[(ny//2+i) * nx + nx//2+j] = 0.2

        elif problem == Problem.NAVIER_STOKES:
            self.channels = 2
            self.initial_u = torch.zeros((nx*ny, self.channels), device="cuda")

        elif problem == Problem.TEST:
            self.channels = 1
            self.initial_u = torch.ones((6, self.channels), device="cuda")
            self.initial_means = self.initial_means[(nx//2-3)*ny+ny//2:(nx//2+3)*ny+ny//2:ny]
            self.initial_scaling = self.initial_scaling[:6]
            self.initial_transforms = self.initial_transforms[:6]

        if problem == Problem.NAVIER_STOKES:
            self.n_boundary_gaussians = 0

            self.boundary_means = torch.empty((self.n_boundary_gaussians, self.d), device="cuda")
            self.boundary_u = torch.empty((self.n_boundary_gaussians, self.channels), device="cuda")
            self.boundary_scaling = torch.empty((self.n_boundary_gaussians, self.d), device="cuda")
            self.boundary_transforms = \
                torch.empty((self.n_boundary_gaussians, self.transform_size), device="cuda")
        
        elif problem == Problem.TEST:
            self.n_boundary_gaussians = 50

            ones = torch.ones(self.n_boundary_gaussians//2, device="cuda") * scale
            boundary_range = torch.linspace(-1,1,self.n_boundary_gaussians//2, device="cuda") * scale
            top_boundary = torch.stack((boundary_range, ones), dim=-1)
            bottom_boundary = torch.stack((boundary_range, -ones), dim=-1)

            self.boundary_means = torch.cat(
                (top_boundary, bottom_boundary))
            self.boundary_u = torch.cat((
                -torch.ones((self.n_boundary_gaussians//2, self.channels), device="cuda"),
                torch.ones((self.n_boundary_gaussians//2, self.channels), device="cuda")
            ), dim=0)
            self.boundary_scaling = torch.ones((self.n_boundary_gaussians, 2), device="cuda") \
                                  / self.n_boundary_gaussians * scale * 1.5
            self.boundary_transforms = \
                torch.zeros((self.n_boundary_gaussians, self.transform_size), device="cuda")

        else:
            self.n_boundary_gaussians = 100

            ones = torch.ones(self.n_boundary_gaussians//4, device="cuda") * scale
            boundary_range = torch.linspace(-1,1,self.n_boundary_gaussians//4, device="cuda") * scale
            left_boundary = torch.stack((-ones, boundary_range), dim=-1)
            right_boundary = torch.stack((ones, boundary_range), dim=-1)
            top_boundary = torch.stack((boundary_range, -ones), dim=-1)
            bottom_boundary = torch.stack((boundary_range, ones), dim=-1)

            self.boundary_means = torch.cat(
                (left_boundary, right_boundary, top_boundary, bottom_boundary))
            self.boundary_u = torch.zeros((self.n_boundary_gaussians, self.channels), device="cuda")
            self.boundary_scaling = torch.ones((self.n_boundary_gaussians, 2), device="cuda") \
                                  / self.n_boundary_gaussians * scale
            self.boundary_transforms = \
                torch.zeros((self.n_boundary_gaussians, self.transform_size), device="cuda")

        self.sampler = GaussianSampler(False)

        def activation(in_dim):
            return nn.Tanh()

        if problem == Problem.NAVIER_STOKES:
            pde_size = 1
        else:
            pde_size = self.channels

        self.dynamics_network = \
            DynamicsNetwork(self.channels, d, pde_size, self.split_size, activation).cuda()

        self.set_initial_params(
            self.initial_means, self.initial_u, self.initial_scaling, self.initial_transforms)

    def randomize(self, n):
        if self.problem == Problem.TEST:
            n = self.initial_means.shape[0]

            if np.random.rand() > 0.75:
                self.initial_means[:,1] = (0.9 + torch.rand(1, device="cuda").repeat(n) * 0.1) \
                                   * ((torch.rand(1, device="cuda").repeat(n) > 0.5) * 2.0 - 1.0)
            else:
                self.initial_means[:,1] = (torch.rand(1, device="cuda").repeat(n) * 2.0 - 1.0) * 0.9

            self.initial_u[:,0] = torch.rand(1, device="cuda").repeat(n) * 2.0 - 1.0

            self.set_initial_params(
                self.initial_means, self.initial_u, self.initial_scaling, self.initial_transforms)
        else:
            tx = torch.linspace(-1, 1, n).cuda() * self.scale
            ty = torch.linspace(-1, 1, n).cuda() * self.scale
            gx, gy = torch.meshgrid((tx,ty), indexing="ij")
            self.initial_means = torch.stack((gx,gy), dim=-1)
            self.initial_means = self.initial_means.reshape(-1, self.d)

            scaling = torch.ones((n*n,self.d), device="cuda") * -4.0
            self.initial_scaling = torch.exp(scaling) * self.scale / (n/20)
            self.initial_transforms = torch.zeros((n*n, self.transform_size), device="cuda")

            sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, self.d, 1)
            # sample_mean = torch.randn(self.d, device="cuda").reshape(1, self.d, 1).clamp(-0.5, 0.5)
            samples = self.initial_means.unsqueeze(-1) - sample_mean
            var = 0.1
            # var = np.log(1.0 + np.random.rand()) * 0.2
            conics = torch.inverse(torch.diag(torch.ones(self.d, device="cuda")) * var * self.scale)
            powers = -0.5 * (samples.transpose(-1, -2) @ (conics @ samples))
            self.initial_u = torch.exp(powers).squeeze(-1) / 3.0
            # self.initial_u = 0.25 * torch.exp(powers).squeeze(-1) / (1.0 + np.random.rand())

            self.initial_boundaries = torch.cat((
                torch.ones(self.n_boundary_gaussians, device="cuda", dtype=torch.bool),
                torch.zeros(n*n, device="cuda", dtype=torch.bool))).unsqueeze(-1)
            self.initial_boundary_mask = ~self.initial_boundaries
            self.initial_boundaries = self.initial_boundaries.to(torch.float32)

            self.reset(False)

            # return

            mask = self.boundary_mask.squeeze(-1)

            if True: # Uniform Noise
                self.means[mask] += torch.randn_like(self.means[mask]) * 0.2
                self.means[mask] = torch.tanh(self.means[mask] / self.scale) * self.scale * 0.95
                self.u[mask] += torch.randn_like(self.u[mask]) * 0.1
            else: # Random
                self.means[mask] = torch.randn_like(self.means[mask])
                self.means[mask] = torch.tanh(self.means[mask] / self.scale) * self.scale * 0.9
                self.u[mask] = torch.randn_like(self.u[mask]) * 0.2

            self.scaling[mask] *= torch.exp(torch.randn_like(self.scaling[mask]) * 0.5)
            self.transforms[mask] = \
                torch.tanh(torch.randn_like(self.transforms[mask]) * 0.3)

            self.full_covariances, self.full_conics = \
                gaussians.build_full_covariances(self.scaling, self.transforms)
            self.covariances, self.conics = \
                gaussians.flatten_covariances(self.full_covariances, self.full_conics)

    def set_initial_params(self, means, u, scaling, transforms, train_initial=False):
        self.true_initial_u = u.clone()
        self.true_initial_means = means.clone()
        self.true_initial_scaling = scaling.clone()
        self.true_initial_transforms = transforms.clone()

        self.initial_u = u # nn.Parameter(u)
        self.initial_means = means # nn.Parameter(means)
        self.initial_scaling = scaling # nn.Parameter(scaling)
        self.initial_transforms = transforms # nn.Parameter(transforms)

        self.boundaries = torch.cat((
            torch.ones(self.n_boundary_gaussians, device="cuda", dtype=torch.bool),
            torch.zeros(means.shape[0], device="cuda", dtype=torch.bool))).unsqueeze(-1)

        self.boundary_mask = ~self.boundaries
        self.boundaries = self.boundaries.to(torch.float32)

        self.initial_boundaries = self.boundaries
        self.initial_boundary_mask = self.boundary_mask

        self.reset(train_initial)

    def reset(self, train_initial=False):
        self.train_initial = train_initial

        self.u = torch.cat((self.boundary_u,
            (self.initial_u if train_initial else self.initial_u.detach()) + 0))
        self.means = torch.cat((self.boundary_means,
            (self.initial_means if train_initial else self.initial_means.detach()) + 0))
        self.scaling = torch.cat((self.boundary_scaling,
              (self.initial_scaling if train_initial else self.initial_scaling.detach()) + 0))
        self.transforms = torch.cat((self.boundary_transforms,
             (self.initial_transforms if train_initial else self.initial_transforms.detach()) + 0))
        self.full_covariances, self.full_conics = \
            gaussians.build_full_covariances(self.scaling, self.transforms)
        self.covariances, self.conics = \
            gaussians.flatten_covariances(self.full_covariances, self.full_conics)

        self.boundaries = self.initial_boundaries
        self.boundary_mask = self.initial_boundary_mask

        self.clear()

    def clear(self):
        self.u_samples = []
        self.bc_u_samples = []
        self.ux_samples = []
        self.uxx_samples = []
        if self.problem == Problem.NAVIER_STOKES:
            self.w_samples = []
            self.wx_samples = []
            self.wxx_samples = []

    def detach(self):
        self.u = self.u.detach()
        self.means = self.means.detach()
        self.scaling = self.scaling.detach()
        self.transforms = self.transforms.detach()
        self.covariances = self.covariances.detach()
        self.conics = self.conics.detach()
        self.full_covariances = self.full_covariances.detach()
        self.full_conics = self.full_conics.detach()

        for i in range(len(self.u_samples)):
            self.u_samples[i] = self.u_samples[i].detach()
            self.ux_samples[i] = self.ux_samples[i].detach()
            self.uxx_samples[i] = self.uxx_samples[i].detach()
            self.bc_u_samples[i] = self.bc_u_samples[i].detach()
            if self.problem == Problem.NAVIER_STOKES:
                self.w_samples[i] = self.w_samples[i].detach()
                self.wx_samples[i] = self.wx_samples[i].detach()
                self.wxx_samples[i] = self.wxx_samples[i].detach()

    def split(self, indices):
        n = indices.sum().item()

        if n == 0:
            return

        # Performing backwards pass through eigen decomposition is unstable
        with torch.no_grad():
            eigvals, eigvecs = torch.linalg.eig(self.full_covariances[indices])
            eigvals, max_idx = torch.max(eigvals.real.abs(), dim=-1, keepdim=True)
            eigvecs = eigvals.unsqueeze(-1) * torch.gather(
                eigvecs.real.transpose(-1, -2), 1, max_idx.unsqueeze(-1).expand(n, 1, self.d))
            displacements = torch.cat((-eigvecs, eigvecs), dim=1)

        split_means = (self.means[indices].reshape(n, 1, self.d) + displacements).reshape(-1, self.d)
        split_scaling = self.scaling[indices].repeat_interleave(self.split_size, 0)
        split_transforms = self.transforms[indices].repeat_interleave(self.split_size, 0)
        split_u = self.u[indices].repeat_interleave(self.split_size, 0) / 2.0

        n = split_means.shape[0]
        self.means = torch.cat((self.means[~indices], split_means), dim=0)
        self.scaling = torch.cat((self.scaling[~indices], split_scaling), dim=0)
        self.transforms = torch.cat((self.transforms[~indices], split_transforms), dim=0)
        self.u = torch.cat((self.u[~indices], split_u), dim=0)
        self.boundaries = torch.cat(
            (self.boundaries[~indices], torch.zeros((n, 1), device="cuda")), dim=0)
        self.boundary_mask = torch.cat(
            (self.boundary_mask[~indices], torch.ones((n,1), dtype=torch.bool, device="cuda")),dim=0)

        self.full_covariances, self.full_conics = \
            gaussians.build_full_covariances(self.scaling, self.transforms)
        self.covariances, self.conics = \
            gaussians.flatten_covariances(self.full_covariances, self.full_conics)

    def pde_rhs(self, samples, u, ux, uxx, wx=None, wxx=None):
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
            return self.nu * (wxx[:,0,0] + wxx[:,1,1]) \
                 - (u[:,0] * wx[:,0] + u[:,1] * wx[:,1])
            # return self.mu * (uxx[:,0,0,:2] + uxx[:,1,1,:2]) \
            #      - (u[:,:2].reshape(-1, 1, 2) * ux[...,:2]).sum(-1) \
            #      - self.inv_rho * ux[...,-1]
            # return self.nu * (uxx[:,0,0] + uxx[:,1,1]) \
            #      - (u.reshape(-1, 1, 2) * ux).sum(-1)

        elif self.problem == Problem.TEST:  
            return torch.zeros_like(u)

        else:
            raise ValueError("Unexpected PDE problem:", self.problem)

    def forward(self, t, dt, split=False):
        with torch.no_grad():
            n = self.means.shape[0]

            self.sampler.preprocess(self.means, self.u, self.covariances, self.conics, self.means)

            sample_u = self.sampler.sample_gaussians() # n, c
            sample_ux = self.sampler.sample_gaussians_derivative() # n, d, c
            sample_uxx = self.sampler.sample_gaussians_laplacian() # n, d, d, c
            if self.problem == Problem.NAVIER_STOKES:
                sample_uxxx = self.sampler.sample_gaussians_third_derivative() # n, d, d, d, c
                sample_wx = sample_uxx[...,0,1] - sample_uxx[...,1,0] # n, d
                sample_wxx = sample_uxxx[...,0,1] - sample_uxxx[...,1,0] # n, d, d
                sample_pde = self.pde_rhs(
                    self.means, sample_u, sample_ux, sample_uxx, sample_wx, sample_wxx
                ).reshape(n, -1)
            else:
                sample_pde = self.pde_rhs(self.means, sample_u, sample_ux, sample_uxx).reshape(n, -1)

            sample_ux = sample_ux.reshape(n, -1)
            sample_uxx = torch.cat((sample_uxx[:,0,0], sample_uxx[:,1,1]), dim=-1).reshape(n, -1)

        self.dynamics_network(
            self.means, self.full_covariances, self.u, self.boundaries,
            sample_u, sample_ux, sample_uxx, sample_pde)

        deltas = self.dynamics_network.compute_deltas(self.sampler)
        self.dmeans = deltas[0]
        mask = self.boundary_mask.squeeze()
        self.dscaling = deltas[1]
        self.dtransforms = deltas[2]
        self.du = deltas[3]

        self.prev_means = self.means.detach().clone()
        self.prev_u = self.u.detach().clone()
        self.prev_scaling = self.scaling.detach().clone()
        self.prev_transforms = self.transforms.detach().clone()
        self.prev_covariances = self.covariances.detach().clone()
        self.prev_conics = self.conics.detach().clone()

        self.means = self.means + self.dmeans * self.boundary_mask
        self.scaling = self.scaling * torch.exp(self.dscaling * self.boundary_mask)
        self.transforms = self.transforms + self.dtransforms * self.boundary_mask
        self.u = self.u + self.du * self.boundary_mask

        if self.problem == Problem.NAVIER_STOKES:
            oob = self.means > 1.0
            self.means[oob] -= 2.0
            oob = self.means < -1.0
            self.means[oob] += 2.0

        self.full_covariances, self.full_conics = \
            gaussians.build_full_covariances(self.scaling, self.transforms)
        self.covariances, self.conics = \
            gaussians.flatten_covariances(self.full_covariances, self.full_conics)

        if not split:
            return

        keep_indices = torch.logical_or(
            torch.norm(torch.abs(self.u), dim=-1) > 0.01, ~self.boundary_mask.squeeze(-1))
        self.u = self.u[keep_indices]
        self.means = self.means[keep_indices]
        self.scaling = self.scaling[keep_indices]
        self.transforms = self.transforms[keep_indices]
        self.covariances = self.covariances[keep_indices]
        self.conics = self.conics[keep_indices]
        self.full_covariances = self.full_covariances[keep_indices]
        self.full_conics = self.full_conics[keep_indices]
        self.boundary_mask = self.boundary_mask[keep_indices]
        self.boundaries = self.boundaries[keep_indices]

        with torch.no_grad():
            n = self.means.shape[0]

            self.sampler.preprocess(self.means, self.u, self.covariances, self.conics, self.means)
            sample_u = self.sampler.sample_gaussians() # n, c
            # sample_ux = self.sampler.sample_gaussians_derivative() # n, d, c
            # sample_uxx = self.sampler.sample_gaussians_laplacian() # n, d, d, c
            # sample_uxxx = self.sampler.sample_gaussians_third_derivative() # n, d, d, c

            # sample_w = sample_ux[:,0,1] - sample_ux[:,1,0]
            # sample_wx = sample_uxx[...,0,1] - sample_uxx[...,1,0]
            # sample_wxx = sample_uxxx[...,0,1] - sample_uxxx[...,1,0]

            ones = torch.ones((n, 1), device="cuda")
            self.sampler.preprocess(self.means, ones, self.covariances, self.conics, self.means)

            density = self.sampler.sample_gaussians() # n, 1
            density = 1.0 - (density - density.min()) / density.max()

            self.sampler.preprocess(self.prev_means, self.prev_u, self.prev_covariances,
                                    self.prev_conics, self.means)
            prev_sample_u = self.sampler.sample_gaussians() # n, c
            # prev_sample_ux = self.sampler.sample_gaussians_derivative() # n, c

            # prev_sample_w = prev_sample_ux[:,0,1] - prev_sample_ux[:,1,0]

            # laplacian = ((sample_uxx[:,0,0] + sample_uxx[:,1,1]) ** 2).unsqueeze(-1)
            # div = (sample_ux[:,0,0] + sample_ux[:,1,1]) ** 2
            # curl = (sample_ux[:,0,1] - sample_ux[:,1,0]) ** 2
            ut = (sample_u - prev_sample_u) ** 2

            # w_laplacian = ((sample_wxx[:,0,0] + sample_wxx[:,1,1]) ** 2).unsqueeze(-1)
            # wt = (sample_w - prev_sample_w) ** 2

            metric = ut * density

            quantile = torch.quantile(metric, 0.98)

            indices = ((metric > quantile).any(-1, keepdim=True) * self.boundary_mask).squeeze(-1)
            print(indices.sum().item())

        # self.means = self.prev_means
        # self.u = self.prev_u
        # self.scaling = self.prev_scaling
        # self.transforms = self.prev_transforms
        # self.covariances = self.prev_covariances
        # self.conics = self.prev_conics

        self.split(indices)

    def sample(self, samples, bc_samples):
        mask = self.boundary_mask.squeeze(-1)
        self.sampler.preprocess(
            self.means[mask], self.u[mask], self.covariances[mask], self.conics[mask], samples)
        u_sample = self.sampler.sample_gaussians() # n, c
        ux = self.sampler.sample_gaussians_derivative() # n, d, c
        uxx = self.sampler.sample_gaussians_laplacian() # n, d, d, c

        self.u_samples.append(u_sample)
        self.ux_samples.append(ux)
        self.uxx_samples.append(uxx)
        if self.problem == Problem.NAVIER_STOKES:
            uxxx = self.sampler.sample_gaussians_third_derivative() # n, d, d, d, c
            self.w_samples.append(ux[:,0,1] - ux[:,1,0])
            self.wx_samples.append(uxx[...,0,1] - uxx[...,1,0])
            self.wxx_samples.append(uxxx[...,0,1] -  uxxx[...,1,0])

        # TODO: Combine with regular samples (cat then split)
        self.sampler.preprocess(
            self.means[mask], self.u[mask], self.covariances[mask], self.conics[mask], bc_samples)
        bc_u_sample = self.sampler.sample_gaussians() # n, c

        self.bc_u_samples.append(bc_u_sample)

    def compute_loss(self, t, dt, samples, time_samples, bc_samples):
        self.sample(samples, bc_samples)
        mask = self.boundary_mask.squeeze()

        if self.rule == IntegrationRule.TRAPEZOID:
            u_sample = time_samples.reshape(-1, 1) * self.u_samples[-1] \
                     + (1 - time_samples.reshape(-1, 1)) * self.u_samples[-2]
            ux = time_samples.reshape(-1, 1, 1) * self.ux_samples[-1] \
               + (1 - time_samples.reshape(-1, 1, 1)) * self.ux_samples[-2]
            uxx = time_samples.reshape(-1, 1, 1, 1) * self.uxx_samples[-1] \
                + (1 - time_samples.reshape(-1, 1, 1, 1)) * self.uxx_samples[-2]
            if self.problem == Problem.NAVIER_STOKES:
                wx = time_samples.reshape(-1, 1) * self.wx_samples[-1] \
                    + (1 - time_samples.reshape(-1, 1)) * self.wx_samples[-2]
                wxx = time_samples.reshape(-1, 1, 1) * self.wxx_samples[-1] \
                    + (1 - time_samples.reshape(-1, 1, 1)) * self.wxx_samples[-2]
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

        if self.problem == Problem.NAVIER_STOKES:
            wt = self.w_samples[-1] - self.w_samples[-2]
        else:
            ut = self.u_samples[-1] - self.u_samples[-2]
        bc_u_sample = self.bc_u_samples[-1]

        pde_loss = torch.zeros(1, device="cuda")
        bc_loss = torch.zeros(1, device="cuda")
        conservation_loss = torch.zeros(1, device="cuda")
        initial_loss = torch.zeros(1, device="cuda")
        magnitude_loss = torch.zeros(1, device="cuda")

        if self.problem == Problem.NAVIER_STOKES:
            rhs = dt * self.pde_rhs(samples, u_sample, ux, uxx, wx, wxx)
        else:
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
            pde_loss += torch.mean((ux[:,0,0] + ux[:,1,1]) ** 2)
            pde_loss += torch.mean((wt - rhs) ** 2)

        elif self.problem == Problem.TEST:
            pde_loss += torch.mean((self.dmeans[mask,1] - self.u[mask,0] / 5.0) ** 2)

        if self.problem == Problem.TEST:
            negative = self.means[mask,1] < -0.8
            if torch.any(negative):
                bc_loss += torch.mean((self.u[mask][negative] - 1.0) ** 2)

            positive = self.means[mask,1] > 0.8
            if torch.any(positive):
                bc_loss += torch.mean((self.u[mask][positive] + 1.0) ** 2)

        elif self.problem != Problem.NAVIER_STOKES:
            bc_loss += torch.mean(bc_u_sample ** 2)

        if self.problem == Problem.TEST:
            conservation_loss += self.dmean_weight * torch.mean(self.dmeans[mask,0] ** 2)
            conservation_loss += self.dmean_weight * torch.mean(
                (self.dmeans[mask] - self.dmeans[mask].mean(0).reshape(1, -1)) ** 2)
            conservation_loss += self.dmean_weight * torch.mean(
                (self.means[mask,1] - self.means[mask,1].mean()) ** 2)
            in_range = self.means[mask,1].abs() < 0.8
            if torch.any(in_range):
                conservation_loss += \
                    self.du_weight * torch.mean((self.u[mask,0][in_range].abs() - 1.0) ** 2)
                conservation_loss += self.du_weight * torch.mean(self.du[mask][in_range] ** 2)
        else:
            conservation_loss += self.dmean_weight * torch.mean(self.dmeans[mask] ** 2)
            conservation_loss += self.du_weight * torch.mean(self.du[mask] ** 2)

        conservation_loss += self.dscale_weight * torch.mean(self.dscaling[mask] ** 2)
        conservation_loss += self.dtransform_weight * torch.mean(self.dtransforms[mask] ** 2)

        if t == 0 and self.train_initial:
            true_initial_covariances, true_initial_conics = gaussians.build_covariances(
                self.true_initial_scaling, self.true_initial_transforms)
            self.sampler.preprocess(self.true_initial_means, self.true_initial_u,
                                    true_initial_covariances, true_initial_conics, samples)
            initial_u_sample = self.sampler.sample_gaussians() # n, c
            initial_loss += torch.mean((self.u_samples[-2] - initial_u_sample) ** 2)

        magnitudes = torch.zeros(ATTENTION_HEADS, device="cuda")
        L = ATTENTION_HEADS*LATENT_SIZE
        for i in range(ATTENTION_HEADS):
            if i == ATTENTION_HEADS-1:
                magnitudes[i] = (self.dynamics_network.local_global_features
                    [...,-L+i*LATENT_SIZE:] ** 2).mean()
            else:
                magnitudes[i] = (self.dynamics_network.local_global_features
                    [...,-L+i*LATENT_SIZE:-L+(i+1)*LATENT_SIZE] ** 2).mean()
        magnitude_loss = ((magnitudes - 1.0) ** 2).mean()

        return self.pde_weight * pde_loss, \
               self.bc_weight * bc_loss, \
               self.conservation_weight * conservation_loss, \
               self.initial_weight * initial_loss, \
               magnitude_loss

    def generate_images(self, res):
        mask = self.boundary_mask.squeeze()
        tx = torch.linspace(-1, 1, res).cuda() * self.scale
        ty = torch.flip(torch.linspace(-1, 1, res).cuda().unsqueeze(-1), (0,1)).squeeze()*self.scale
        gx, gy = torch.meshgrid((tx, ty), indexing="xy")
        samples = torch.stack((gx, gy), dim=-1).reshape(res * res, self.d)

        self.sampler.preprocess(self.means[mask], self.u[mask], self.covariances[mask], self.conics[mask], samples)

        u = self.sampler.sample_gaussians().reshape(res * res, -1).detach().cpu().numpy()

        return u.transpose(0, 1).reshape(-1, res, res)

    def plot_gaussians(self):
        return gaussians.plot_gaussians(self.means, self.covariances, self.u, self.scale)
