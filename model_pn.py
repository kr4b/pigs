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

class RBFAct(nn.Module):
    def __init__(self, in_dim):
        super(RBFAct, self).__init__() 
        self.b = nn.Parameter(torch.ones(1), requires_grad=True)
        self.c = nn.Parameter(torch.zeros(in_dim), requires_grad=True)

    def forward(self, x):
        return torch.exp(-self.b * (x - self.c) ** 2)

# LATENT_SIZE = 512
# L1_SIZE = 64
# L2_SIZE = 128
# L3_SIZE = 256
LATENT_SIZE = 64
L1_SIZE = 16
L2_SIZE = 32
L3_SIZE = 48
EMBEDDING_SIZE = 21

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
    def __init__(self, c, d, pde_size, activation):
        super(InputTransform, self).__init__()
        self.c = c
        self.d = d
        self.pde_size = pde_size

        self.latent_net = LatentTransform(d + d * d + 2*c + 2*d*c, activation)
        self.transform_net = TransformNet(d, activation)
        self.transform_u_net = TransformNet(c, activation)
        self.transform_ux_net = TransformNet(d*c, activation)
        self.transform_uxx_net = TransformNet(d*c, activation)
        self.transform_pde_net = TransformNet(pde_size, activation)

    def forward(self, means, full_covariances, u, sample_u, sample_ux, sample_uxx, sample_pde):
        means = means.unsqueeze(0) # 1, n, d
        covariances = full_covariances.reshape(1, -1, self.d*self.d) # 1, n, d*d
        u = u.unsqueeze(0) # 1, n, c
        sample_u = sample_u.unsqueeze(0) # 1, n, c
        ux = sample_ux.unsqueeze(0) # 1, n, d*c
        uxx = sample_uxx.unsqueeze(0) # 1, n, d*c
        pde = sample_pde.unsqueeze(0) # 1, n, pde_size
        params = torch.cat((means, covariances, u, sample_u, ux, uxx), dim=-1).transpose(1, 2)

        latent = self.latent_net(params).mean(-1) # 1, LATENT_SIZE
        # latent = torch.cat((latent, feature), dim=-1)
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

class FeatureTransform(nn.Module):
    def __init__(self, activation):
        super(FeatureTransform, self).__init__()

        self.latent_net = LatentTransform(L1_SIZE, activation)
        self.transform_net = TransformNet(L1_SIZE, activation)

    def forward(self, x, feature):
        latent = self.latent_net(x).mean(-1) # 1, LATENT_SIZE
        latent = torch.cat((latent, feature), dim=-1)
        self.transform = self.transform_net(latent)
        return self.transform @ x

    def transform_features(self, x):
        return self.transform @ x

def delta_network(m, c, d, activation):
    transform_size = d * (d-1) // 2
    return nn.Sequential(
        nn.Linear(2 * LATENT_SIZE, LATENT_SIZE),
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

        self.input_transform = InputTransform(c, d, pde_size, activation)
        self.input_projection = nn.Sequential(
            nn.Linear(d * d + 2*c + 2*d*c + pde_size, L1_SIZE),
            # nn.LayerNorm(L1_SIZE),
            activation(L1_SIZE),
            nn.Linear(L1_SIZE, L2_SIZE),
            # nn.LayerNorm(L2_SIZE),
            activation(L2_SIZE),
            nn.Linear(L2_SIZE, LATENT_SIZE),
        )
        self.distance_transform = nn.Parameter(
            torch.rand((LATENT_SIZE, EMBEDDING_SIZE), device="cuda")
        )
        self.transform = nn.Sequential(
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            activation(LATENT_SIZE),
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            activation(LATENT_SIZE),
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            activation(LATENT_SIZE),
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
        )
        self.query_transform = nn.Sequential(
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            activation(LATENT_SIZE),
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            activation(LATENT_SIZE),
            nn.Linear(LATENT_SIZE, (LATENT_SIZE+L1_SIZE)//2),
            activation((LATENT_SIZE+L1_SIZE)//2),
            nn.Linear((LATENT_SIZE+L1_SIZE)//2, L1_SIZE),
        )
        self.key_transform = nn.Sequential(
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            activation(LATENT_SIZE),
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            activation(LATENT_SIZE),
            nn.Linear(LATENT_SIZE, (LATENT_SIZE+L1_SIZE)//2),
            activation((LATENT_SIZE+L1_SIZE)//2),
            nn.Linear((LATENT_SIZE+L1_SIZE)//2, L1_SIZE),
        )
        self.frequencies = nn.Parameter(
            torch.randn((EMBEDDING_SIZE-1) // d // 2, device="cuda") * 10,
            requires_grad=False
        )
        self.delta_net = delta_network(1, c, d, activation)
        # self.split_delta_net = delta_network(split_size, c, d, activation)

    def forward(self, means, full_covariances, u, sample_u, sample_ux, sample_uxx, sample_pde):
        t_means, t_covariances, t_u, t_sample_u, t_ux, t_uxx, t_pde = \
            self.input_transform(means, full_covariances, u, sample_u, sample_ux, sample_uxx, sample_pde)

        t_means = t_means.reshape(1, -1, self.d)
        t_covariances = t_covariances.reshape(1, -1, self.d*self.d)
        t_u = t_u.reshape(1, -1, self.c)
        t_sample_u = t_sample_u.reshape(1, -1, self.c)
        t_ux = t_ux.reshape(1, -1, self.d*self.c)
        t_uxx = t_uxx.reshape(1, -1, self.d*self.c)
        t_pde = t_pde.reshape(1, -1, self.pde_size)
        t_params = torch.cat((t_covariances, t_u, t_sample_u, t_ux, t_uxx, t_pde), dim=-1)

        self.global_features = self.input_projection(t_params) # 1, n, LATENT_SIZE

    def _compute_deltas(self, sampler, features, delta_net, size):
        b, n, l = features.shape

        transforms = self.transform(features).reshape(b, n, l) # 1, n, LATENT_SIZE
        queries = self.query_transform(features).reshape(b, n, -1) # 1, n, L1_SIZE
        keys = self.key_transform(features).reshape(b, n, -1) # 1, n, L1_SIZE
        indices, neighbor_features1 = sampler.aggregate_neighbors(
            features.squeeze(0), transforms.squeeze(0), queries.squeeze(0),
            keys.squeeze(0), self.frequencies, self.distance_transform)
        # if torch.isnan(neighbor_features1.mean()):
        #     for i in range(neighbor_features1.shape[0]):
        #         if torch.isnan(neighbor_features1[i].mean()):
        #             print(i, neighbor_features1[i])
        # distance_transforms = \
        #     self.distance_transform(combined_features).reshape(b, n, l, -1) # 1, n, LATENT_SIZE, -1
        # _, neighbor_features2 = \
        #     sampler.aggregate_neighbors(neighbor_features1, distance_transforms.squeeze(0))

        local_global_features = torch.cat((
            features,
            neighbor_features1.unsqueeze(0),
        ), dim=-1)

        deltas = delta_net(local_global_features)
        dmeans = deltas[...,:self.d*size].squeeze(0)
        dscaling = deltas[...,self.d*size:2*self.d*size].squeeze(0)
        dtransforms = deltas[...,2*self.d*size:(2*self.d+self.transform_size)*size].squeeze(0)
        du = deltas[...,-self.c*size:].squeeze(0)

        return dmeans, dscaling, dtransforms, du

    def compute_deltas(self, sampler):
        features = self.global_features # 1, n, LATENT_SIZE
        return self._compute_deltas(sampler, features, self.delta_net, 1)

    def split(self, sampler, indices):
        features = self.global_features[:,indices,:]\
                       .reshape(-1, indices.shape[0], LATENT_SIZE) # 1, n, LATENT_SIZE
        return self._compute_deltas(sampler, features, self.split_delta_net, self.split_size)

    def transform_gaussians(self, means, full_covariances, u, sample_u, sample_ux, sample_uxx, sample_pde):
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
        t_params = torch.cat((t_covariances, t_u, t_sample_u, t_ux, t_uxx, t_pde), dim=-1)

        features = self.input_projection(t_params) # 1, n, L1_SIZE
        self.global_features = torch.cat((self.global_features, features), dim=-2)


class Model(nn.Module):
    def __init__(self, problem, rule, nx, ny, d, scale):
        super(Model, self).__init__()
        self.problem = problem
        self.rule = rule
        self.nx = nx
        self.ny = ny
        self.d = d
        self.scale = scale

        self.pde_weight = 1.0
        self.bc_weight = 1.0
        self.conservation_weight = 0.1
        self.initial_weight = 2.0

        self.split_size = 4

        if problem == Problem.BURGERS:
            self.nu = 1.0 / (100.0 * np.pi)
        elif problem == Problem.NAVIER_STOKES:
            self.nu = 1e-3

        tx = torch.linspace(-1, 1, nx).cuda() * scale
        ty = torch.linspace(-1, 1, ny).cuda() * scale
        gx, gy = torch.meshgrid((tx,ty), indexing="ij")
        self.initial_means = torch.stack((gx,gy), dim=-1)# * 0
        self.initial_means = self.initial_means.reshape(-1, d)

        scaling = torch.ones((nx*ny,d), device="cuda") * -5.5
        self.initial_scaling = torch.exp(scaling) * scale

        self.transform_size = d * (d - 1) // 2
        self.initial_transforms = torch.zeros((nx*ny,self.transform_size), device="cuda")

        if problem == Problem.BURGERS or problem == Problem.DIFFUSION:
            self.channels = 1

            sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, d, 1)
            samples = self.initial_means.unsqueeze(-1) - sample_mean
            conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1 * scale)
            powers = -0.5 * (samples.transpose(-1, -2) @ (conics @ samples))
            self.initial_u = torch.exp(powers).squeeze(-1) / 5.0

        elif problem == Problem.WAVE:
            self.channels = 2
            self.initial_u = torch.zeros((nx*ny, self.channels), device="cuda")
            for i in range(-2, 3):
                for j in range(-2, 3):
                    self.initial_u[(ny//2+i) * nx + nx//2+j] = 0.2

        elif problem == Problem.NAVIER_STOKES:
            self.channels = 1
            self.initial_u = torch.zeros((nx*ny, self.channels), device="cuda")

        self.sampler = GaussianSampler(False)

        def activation(in_dim):
            return nn.Tanh()
            # return RBFAct(in_dim)

        self.dynamics_network = \
            DynamicsNetwork(self.channels, d, self.channels, self.split_size, activation).cuda()

        self.set_initial_params(
            self.initial_means, self.initial_u, self.initial_scaling, self.initial_transforms)

    def set_initial_params(self, means, u, scaling, transforms):
        self.true_initial_u = u.clone()
        self.true_initial_means = means.clone()
        self.true_initial_scaling = scaling.clone()
        self.true_initial_transforms = transforms.clone()

        self.initial_u = nn.Parameter(u)
        self.initial_means = nn.Parameter(means)
        self.initial_scaling = nn.Parameter(scaling)
        self.initial_transforms = nn.Parameter(transforms)

        self.reset()

    def reset(self):
        self.u = self.initial_u.detach() + 0
        self.means = self.initial_means.detach() + 0
        self.scaling = self.initial_scaling.detach() + 0
        self.transforms = self.initial_transforms.detach() + 0
        self.full_covariances, self.full_conics = \
            gaussians.build_full_covariances(self.scaling,self.transforms)
        self.covariances, self.conics = \
            gaussians.flatten_covariances(self.full_covariances, self.full_conics)

        self.clear()

    def clear(self):
        self.u_samples = []
        self.bc_u_samples = []
        self.ux_samples = []
        self.uxx_samples = []

    def detach(self):
        self.u = self.u.detach()
        self.means = self.means.detach()
        self.scaling = self.scaling.detach()
        self.transforms = self.transforms.detach()
        self.covariances = self.covariances.detach()
        self.conics = self.conics.detach()
        self.full_covariances = self.full_covariances.detach()
        self.full_conics = self.full_conics.detach()

    def split(self, indices):
        self.sampler.preprocess(self.means[indices], self.u[indices], self.covariances[indices], self.conics[indices], self.means[indices])
        dmeans, dscaling, dtransforms, du = self.dynamics_network.split(self.sampler, indices)

        # Performing backwards pass through eigen decomposition is unstable
        with torch.no_grad():
            eigvals, eigvecs = torch.linalg.eig(self.full_covariances[indices])
            # eigvecs = eigvecs.real * eigvals.real
            # torch.tensor([-eigvecs[0], eigvecs[0], -eigvecs[1], eigvecs[1]], device="cuda")

        n = indices.shape[0]

        split_means = (self.means[indices].repeat(1, self.split_size).reshape(n, -1, self.d) \
                    + 3 * eigvals.real.amax(-1).reshape(n, 1, 1)
                    * dmeans.reshape(n, -1, self.d)).reshape(-1, self.d)
        half_scale = self.scaling[indices].repeat(1, self.split_size).reshape(n, -1, self.d) * 0.5
        split_scaling = (half_scale * torch.exp(dscaling.reshape(n, -1, self.d))).reshape(-1, self.d)
        split_transforms = \
            (self.transforms[indices].repeat(1, self.split_size).reshape(n, -1, self.transform_size)\
            + dtransforms.reshape(n, -1, self.transform_size)).reshape(-1, self.transform_size)
        split_u = du.reshape(-1, self.channels)

        split_full_covariances, split_full_conics = \
            gaussians.build_full_covariances(split_scaling, split_transforms)
        split_covariances, split_conics = \
            gaussians.flatten_covariances(split_full_covariances, split_full_conics)

        with torch.no_grad():
            n = split_means.shape[0]

            self.sampler.preprocess(
                split_means, split_u, split_covariances, split_conics, split_means)

            sample_u = self.sampler.sample_gaussians() # n, c
            sample_ux = self.sampler.sample_gaussians_derivative() # n, d, c
            sample_uxx = self.sampler.sample_gaussians_laplacian() # n, d, d, c
            sample_pde = self.pde_rhs(split_means, sample_u, sample_ux, sample_uxx).reshape(n, -1)
            # sample_pde = torch.zeros((n, self.channels), device="cuda")

            sample_ux = sample_ux.reshape(n, -1)
            sample_uxx = torch.stack((sample_uxx[:,0,0], sample_uxx[:,1,1]), dim=-2).reshape(n, -1)

        self.dynamics_network.transform_gaussians(
            split_means, split_full_covariances, split_u, sample_u, sample_ux, sample_uxx,sample_pde)

        self.means = torch.cat((self.means, split_means), dim=0)
        self.scaling = torch.cat((self.scaling, split_scaling), dim=0)
        self.transforms = torch.cat((self.transforms, split_transforms), dim=0)
        self.covariances = torch.cat((self.covariances, split_covariances), dim=0)
        self.conics = torch.cat((self.conics, split_conics), dim=0)
        self.u = torch.cat((self.u, split_u), dim=0)

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
            return 0
            # return self.mu * (uxx[:,0,0,:2] + uxx[:,1,1,:2]) \
            #      - (u[:,:2].reshape(-1, 1, 2) * ux[...,:2]).sum(-1) \
            #      - self.inv_rho * ux[...,-1]
            # return self.nu * (uxx[:,0,0] + uxx[:,1,1]) \
            #      - (u.reshape(-1, 1, 2) * ux).sum(-1)

        else:
            raise ValueError("Unexpected PDE problem:", self.problem)

    def forward(self, t, dt):
        with torch.no_grad():
            n = self.means.shape[0]

            self.sampler.preprocess(self.means, self.u, self.covariances, self.conics, self.means)

            sample_u = self.sampler.sample_gaussians() # n, c
            sample_ux = self.sampler.sample_gaussians_derivative() # n, d, c
            sample_uxx = self.sampler.sample_gaussians_laplacian() # n, d, d, c
            sample_pde = self.pde_rhs(self.means, sample_u, sample_ux, sample_uxx).reshape(n, -1)
            # sample_pde = torch.zeros((n, self.channels), device="cuda")

            sample_u = sample_u.reshape(n, -1)
            sample_ux = sample_ux.reshape(n, -1)
            sample_uxx = torch.stack((sample_uxx[:,0,0], sample_uxx[:,1,1]), dim=-2).reshape(n, -1)

        self.dynamics_network(
            self.means, self.full_covariances, self.u, sample_u, sample_ux, sample_uxx, sample_pde)

        # TODO: Split
        # self.split(torch.tensor([0], dtype=torch.long, device="cuda"))
        # self.sampler.preprocess(self.means, self.u, self.covariances, self.conics, self.means)

        deltas = self.dynamics_network.compute_deltas(self.sampler)
        self.dmeans = deltas[0]
        self.dscaling = deltas[1]
        self.dtransforms = deltas[2]
        self.du = deltas[3]

        # self.means = self.means + self.dmeans
        self.scaling = self.scaling * torch.exp(self.dscaling)
        self.transforms = self.transforms + self.dtransforms
        self.u = self.u + self.du

        self.full_covariances, self.full_conics = \
            gaussians.build_full_covariances(self.scaling,self.transforms)
        self.covariances, self.conics = \
            gaussians.flatten_covariances(self.full_covariances, self.full_conics)

    def sample(self, samples, bc_samples):
        self.sampler.preprocess(self.means, self.u, self.covariances, self.conics, samples)
        u_sample = self.sampler.sample_gaussians() # n, c
        ux = self.sampler.sample_gaussians_derivative() # n, d, c
        uxx = self.sampler.sample_gaussians_laplacian() # n, d, d, c

        # TODO: Combine with regular samples (cat then split)
        self.sampler.preprocess(self.means, self.u, self.covariances, self.conics, bc_samples)
        bc_u_sample = self.sampler.sample_gaussians() # n, c

        self.u_samples.append(u_sample)
        self.bc_u_samples.append(bc_u_sample)
        self.ux_samples.append(ux)
        self.uxx_samples.append(uxx)

    def compute_loss(self, t, dt, samples, time_samples, bc_samples):
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
        initial_loss = torch.zeros(1, device="cuda")

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

        # elif self.problem == Problem.NAVIER_STOKES:
            # self.sampler.preprocess(
            #     self.prev_means, self.translation, self.prev_covariances, self.prev_conics, samples)
            # translation_sample = self.sampler.sample_gaussians() # n, c

            # bc_mask = self.bc_mask(samples)
            # pde_loss += torch.mean(bc_mask * (translation_sample - self.u_samples[-2][...,:2]) ** 2)
            # pde_loss += torch.mean(bc_mask * (ux[:,0,0] + ux[:,1,1]) ** 2)
            # pde_loss += torch.mean(bc_mask * (ut[...,:2] - rhs) ** 2)

            # pde_loss += torch.mean((ux[:,0,0] + ux[:,1,1]) ** 2)
            # pde_loss += torch.mean((ut[...,:2] - rhs) ** 2)

        bc_loss += torch.mean(bc_u_sample ** 2)
        conservation_loss += torch.mean(self.du ** 2)
        conservation_loss += torch.mean(self.dmeans ** 2)
        conservation_loss += torch.mean(self.dtransforms ** 2)
        conservation_loss += torch.mean(self.dscaling ** 2)

        if t == 0:
            true_initial_covariances, true_initial_conics = gaussians.build_covariances(
                self.true_initial_scaling, self.true_initial_transforms)
            self.sampler.preprocess(self.true_initial_means, self.true_initial_u,
                                    true_initial_covariances, true_initial_conics, samples)
            initial_u_sample = self.sampler.sample_gaussians() # n, c
            initial_loss += torch.mean((self.u_samples[-2] - initial_u_sample) ** 2)

        return self.pde_weight * pde_loss, \
               self.bc_weight * bc_loss, \
               self.conservation_weight * conservation_loss, \
               self.initial_weight * initial_loss

    def generate_images(self, res):
        if self.problem == Problem.WAVE:
            img1 = gaussians.sample_gaussians_img(
                self.means, self.full_conics, self.u[...,0], res, res, self.scale
            ).detach().cpu().numpy()

            img2 = gaussians.sample_gaussians_img(
                self.means, self.full_conics, self.u[...,1], res, res, self.scale
            ).detach().cpu().numpy()

            return np.stack([img1, img2])
        else:
            img1 = gaussians.sample_gaussians_img(
                self.means, self.full_conics, self.u, res, res, self.scale
            ).detach().cpu().numpy()

            return img1

    def plot_gaussians(self):
        return gaussians.plot_gaussians(self.means, self.covariances, self.u, self.scale)
