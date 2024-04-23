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

# LATENT_SIZE = 512
# L1_SIZE = 64
# L2_SIZE = 128
# L3_SIZE = 256
LATENT_SIZE = 64
L1_SIZE = 16
L2_SIZE = 32
L3_SIZE = 32
DISTANCE_EMBEDDINGS = 5

class LatentTransform(nn.Module):
    def __init__(self, in_channels, activation):
        super(LatentTransform, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, L1_SIZE, 1),
            # nn.BatchNorm1d(L1_SIZE),
            activation,
            nn.Conv1d(L1_SIZE, L2_SIZE, 1),
            # nn.BatchNorm1d(L2_SIZE),
            activation,
            nn.Conv1d(L2_SIZE, LATENT_SIZE, 1),
            # nn.BatchNorm1d(LATENT_SIZE),
            activation,
        )

    def forward(self, x):
        return self.layers(x)

class TransformNet(nn.Module):
    def __init__(self, d, activation):
        super(TransformNet, self).__init__()
        self.d = d
        self.layers = nn.Sequential(
            nn.Linear(LATENT_SIZE, L3_SIZE),
            # nn.BatchNorm1d(L2_SIZE),
            activation,
            nn.Linear(L3_SIZE, L2_SIZE),
            # nn.BatchNorm1d(L2_SIZE),
            activation,
            nn.Linear(L2_SIZE, d*d),
        )

    def forward(self, x):
        return torch.eye(self.d, device="cuda").unsqueeze(0) \
             + self.layers(x).reshape(-1, self.d, self.d) # 1, d, d

class InputTransform(nn.Module):
    def __init__(self, c, d, activation):
        super(InputTransform, self).__init__()
        self.c = c
        self.d = d

        self.latent_net = LatentTransform(d + d * d + 2*c + 2*d*c, activation)
        self.transform_net = TransformNet(d, activation)
        self.transform_u_net = TransformNet(c, activation)
        self.transform_ux_net = TransformNet(d*c, activation)
        self.transform_uxx_net = TransformNet(d*c, activation)

    def forward(self, means, covariances, u, sample_u, sample_ux, sample_uxx):
        means = means.unsqueeze(0) # 1, n, d
        covariances = covariances.reshape(1, -1, self.d*self.d) # 1, n, d*d
        u = u.unsqueeze(0) # 1, n, c
        sample_u = sample_u.unsqueeze(0) # 1, n, c
        ux = sample_ux.unsqueeze(0) # 1, n, d*c
        uxx = sample_uxx.unsqueeze(0) # 1, n, d*c
        params = torch.cat((means, covariances, u, sample_u, ux, uxx), dim=-1).transpose(1, 2)

        latent = self.latent_net(params).mean(-1) # 1, LATENT_SIZE
        # latent = torch.cat((latent, feature), dim=-1)
        self.transform = self.transform_net(latent) # 1, d, d
        self.transform_u = self.transform_u_net(latent) # 1, c, c
        self.transform_ux = self.transform_ux_net(latent) # 1, d*c, d*c
        self.transform_uxx = self.transform_uxx_net(latent) # 1, d*c, d*c

        covariances = covariances.reshape(-1, self.d, self.d)
        means = means.reshape(-1, self.d, 1)
        u = u.reshape(-1, self.c, 1)
        sample_u = sample_u.reshape(-1, self.c, 1)
        ux = ux.reshape(-1, self.d*self.c, 1)
        uxx = uxx.reshape(-1, self.d*self.c, 1)
        return self.transform @ means, \
               self.transform @ covariances, \
               self.transform_u @ u, \
               self.transform_u @ sample_u, \
               self.transform_ux @ ux, \
               self.transform_uxx @ uxx

    def transform_gaussians(self, means, covariances, u):
        covariances = covariances.reshape(-1, self.d, self.d)
        means = means.reshape(-1, self.d, 1)
        u = u.reshape(-1, self.c, 1)
        return self.transform @ means, self.transform @ covariances, self.transform_u @ u

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
        # nn.BatchNorm1d(LATENT_SIZE),
        activation,
        # nn.Linear(L3_SIZE, L2_SIZE),
        # nn.BatchNorm1d(L2_SIZE),
        # activation,
        nn.Linear(LATENT_SIZE, L2_SIZE),
        # nn.BatchNorm1d(L2_SIZE),
        activation,
        nn.Linear(L2_SIZE, m * (d + d + transform_size + c)),
    )

class DynamicsNetwork(nn.Module):
    def __init__(self, c, d, activation):
        super(DynamicsNetwork, self).__init__()
        self.c = c
        self.d = d
        self.transform_size = d * (d-1) // 2

        self.input_transform = InputTransform(c, d, activation)
        self.input_projection = nn.Sequential(
            nn.Linear(d * d + 2*c + 2*d*c, L1_SIZE),
            # nn.BatchNorm1d(L1_SIZE),
            activation,
            nn.Linear(L1_SIZE, L2_SIZE),
            # nn.BatchNorm2d(L2_SIZE),
            activation,
            nn.Linear(L2_SIZE, LATENT_SIZE),
        )
        self.distance_transform = nn.Sequential(
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            # nn.BatchNorm1d(((DISTANCE_EMBEDDINGS*d + 1)*LATENT_SIZE)//2),
            activation,
            nn.Linear(LATENT_SIZE, (DISTANCE_EMBEDDINGS*d + 1)*LATENT_SIZE),
            # nn.BatchNorm1d((DISTANCE_EMBEDDINGS*d + 1)*LATENT_SIZE),
            # activation,
            # nn.Linear(d*LATENT_SIZE, d*LATENT_SIZE),
            # nn.BatchNorm1d(d*LATENT_SIZE),
        )
        self.delta_net = delta_network(1, c, d, activation)

    def forward(self, means, covariances, u, sample_u, sample_ux, sample_uxx):
        n, _ = means.shape

        t_means, t_covariances, t_u, t_sample_u, t_ux, t_uxx = \
            self.input_transform(means, covariances, u, sample_u, sample_ux, sample_uxx)

        t_means = t_means.reshape(1, -1, self.d)
        t_covariances = t_covariances.reshape(1, -1, self.d*self.d)
        t_u = t_u.reshape(1, -1, self.c)
        t_sample_u = t_sample_u.reshape(1, -1, self.c)
        t_ux = t_ux.reshape(1, -1, self.d*self.c)
        t_uxx = t_uxx.reshape(1, -1, self.d*self.c)
        t_params = torch.cat((t_covariances, t_u, t_sample_u, t_ux, t_uxx), dim=-1)

        self.global_features = self.input_projection(t_params).transpose(1, 2) # 1, n, LATENT_SIZE

    def compute_deltas(self, sampler):
        b, _, n = self.global_features.shape
        e = DISTANCE_EMBEDDINGS*self.d+1

        features = self.global_features.transpose(1, 2) # 1, n, LATENT_SIZE

        distance_transforms = \
            self.distance_transform(features).reshape(b, n, -1, e) # 1, n, LATENT_SIZE, e
        _, neighbor_features = \
            sampler.aggregate_neighbors(features.squeeze(), distance_transforms.squeeze())
        distance_transforms = \
            self.distance_transform(neighbor_features).reshape(b, n, -1, e) # 1, n, LATENT_SIZE, e
        _, neighbor_features = \
            sampler.aggregate_neighbors(neighbor_features.squeeze(), distance_transforms.squeeze())

        local_global_features = torch.cat((
            features,
            neighbor_features.unsqueeze(0),
        ), dim=-1)

        deltas = self.delta_net(local_global_features)
        dmeans = deltas[...,:self.d].squeeze(0)
        dscaling = deltas[...,self.d:2*self.d].squeeze(0)
        dtransforms = deltas[...,2*self.d:2*self.d+self.transform_size].squeeze(0)
        du = deltas[...,-self.c:].squeeze(0)

        return dmeans, dscaling, dtransforms, du

    def transform_gaussians(self, means, covariances, u):
        t_means, t_covariances, t_u = \
            self.input_transform.transform_gaussians(means, covariances, u)

        t_means = t_means.reshape(1, -1, self.d)
        t_covariances = t_covariances.reshape(1, -1, self.d*self.d)
        t_u = t_u.reshape(1, -1, self.c)
        t_params = torch.cat((t_covariances, t_u), dim=-1)

        features = self.input_projection(t_params) # 1, n, L1_SIZE
        features = self.feature_transform.transform_features(features.transpose(1, 2)) # 1,L1_SIZE,n
        self.features = torch.cat((self.features, features), dim=-1)

        #global_feature = self.feature_projection(features).mean(-1) # 1, LATENT_SIZE
        #self.global_gaussians_feature = torch.mean(self.global_gaussians_feature, global_feature)

class SplitNetwork(nn.Module):
    def __init__(self, c, d, split_size, activation):
        super(SplitNetwork, self).__init__()
        self.c = c
        self.d = d
        self.transform_size = d * (d-1) // 2
        self.split_size = split_size

        self.delta_net = delta_network(self.split_size, c, d, activation)

    def forward(self, feature, global_feature):
        local_global_feature = torch.cat((feature, global_feature), dim=-1)
        deltas = self.delta_net(local_global_feature)

        dscaling = deltas[...,:self.d*self.split_size].squeeze(0)

        start = self.d*self.split_size
        end = start+self.transform_size*self.split_size
        dtransforms = deltas[...,start:end].squeeze(0)

        du = deltas[...,-self.c*self.split_size:].squeeze(0)

        return dscaling, dtransforms, du

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

        self.split_size = 4

        self.nu = 1.0 / (100.0 * np.pi)

        tx = torch.linspace(-1, 1, nx).cuda() * scale
        ty = torch.linspace(-1, 1, ny).cuda() * scale
        gx, gy = torch.meshgrid((tx,ty), indexing="ij")
        self.initial_means = torch.stack((gx,gy), dim=-1)
        self.initial_means = self.initial_means.reshape(-1, d)

        scaling = torch.ones((nx*ny,d), device="cuda") * -4.0
        self.initial_scaling = torch.exp(scaling) * scale

        self.transform_size = d * (d - 1) // 2
        self.initial_transform = torch.zeros((nx*ny,self.transform_size), device="cuda")

        self.initial_covariances = \
            gaussians.build_covariances(self.initial_scaling, self.initial_transform)
        self.initial_conics = torch.inverse(self.initial_covariances)

        if problem == Problem.BURGERS or problem == Problem.DIFFUSION:
            self.channels = 1

            sample_mean = torch.tensor([0.0, 0.0], device="cuda").reshape(1, d, 1)
            samples = self.initial_means.unsqueeze(-1) - sample_mean
            conics = torch.inverse(torch.diag(torch.ones(d, device="cuda")) * 0.1 * self.scale)
            powers = -0.5 * (samples.transpose(-1, -2) @ (conics @ samples))
            self.initial_u = torch.exp(powers).squeeze(-1) / 4.0

        elif problem == Problem.WAVE:
            self.channels = 2
            self.initial_u = torch.zeros((nx*ny, 2), device="cuda")
            for i in range(-2, 3):
                for j in range(-2, 3):
                    self.initial_u[(ny//2+i) * nx + nx//2+j] = 0.2

        self.sampler = GaussianSampler(False)
        self.dynamics_network = DynamicsNetwork(self.channels, d, nn.Tanh()).cuda()
        # self.split_network = SplitNetwork(self.channels, d, self.split_size, nn.Tanh()).cuda()

        self.reset()

    def reset(self):
        self.u = self.initial_u
        self.means = self.initial_means
        self.scaling = self.initial_scaling
        self.transforms = self.initial_transform
        self.covariances = self.initial_covariances
        self.conics = self.initial_conics

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

    def split(self, indices, scale):
        feature = self.dynamics_network.features[indices]
        global_feature = self.dynamics_network.global_feature
        dscaling, dtransforms, du = self.split_network(feature, global_feature)

        eigvals, eigvecs = torch.linalg.eig(self.covariances[indices])
        eigvecs = eigvecs * eigvals

        split_means = self.means[indices].unsqueeze(0) + \
                      torch.tensor([-eigvecs[0], eigvecs[0], -eigvecs[1], eigvecs[1]], device="cuda")
        split_scaling = self.scaling[indices].unsqueeze(0) / 2.0 + dscaling
        split_transforms = self.scaling[indices].unsqueeze(0) + dtransforms
        split_covariances = gaussians.build_covariances(split_scaling, split_transforms)
        split_conics = torch.inverse(split_covariances)
        split_u = du

        self.dynamics_network.transform_gaussians(split_means, split_covariances, split_u)

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
            return self.mu * (uxx[:,0,0,:2] + uxx[:,1,1,:2]) \
                 - (u[:,:2].reshape(-1, 1, 2) * ux[...,:2]).sum(-1) \
                 - self.inv_rho * ux[...,-1]

        else:
            raise ValueError("Unexpected PDE problem:", self.problem)

    def forward(self, dt):
        self.covariances = gaussians.build_covariances(self.scaling, self.transforms)
        self.conics = torch.inverse(self.covariances)

        n = self.means.shape[0]

        with torch.no_grad():
            self.sampler.preprocess(self.means, self.u, self.covariances, self.conics, self.means)

            sample_u = self.sampler.sample_gaussians() # n, c
            sample_ux = self.sampler.sample_gaussians_derivative() # n, d, c
            sample_uxx = self.sampler.sample_gaussians_laplacian() # n, d, d, c
            sample_pde = self.pde_rhs(self.means, sample_u, sample_ux, sample_uxx).reshape(n, -1)

            sample_u = sample_u.reshape(n, -1)
            sample_ux = sample_ux.reshape(n, -1)
            sample_uxx = torch.stack((sample_uxx[:,0,0], sample_uxx[:,1,1]), dim=-2).reshape(n, -1)

        self.dynamics_network(
            self.means, self.covariances, self.u, sample_u, sample_ux, sample_uxx)

        # TODO: Split

        deltas = self.dynamics_network.compute_deltas(self.sampler)
        self.dmeans = deltas[0]
        self.dscaling = deltas[1]
        self.dtransforms = deltas[2]
        self.du = deltas[3]

        self.means = self.means + self.dmeans
        self.scaling = self.scaling * torch.exp(self.dscaling)
        self.transforms = self.transforms + self.dtransforms
        self.u = self.u + self.du

    def sample(self, samples, bc_samples):
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
        conservation_loss += torch.mean(self.du ** 2)
        conservation_loss += torch.mean(self.dmeans ** 2)
        conservation_loss += torch.mean(self.dtransforms ** 2)
        conservation_loss += torch.mean(self.dscaling ** 2)

        return self.pde_weight * pde_loss,\
               self.bc_weight * bc_loss,\
               self.conservation_weight * conservation_loss

    def generate_images(self, res):
        if self.problem == Problem.BURGERS or self.problem == Problem.DIFFUSION:
            img1 = gaussians.sample_gaussians_img(
                self.means, self.conics, self.u, res, res, self.scale
            ).detach().cpu().numpy()

            return img1
        elif self.problem == Problem.WAVE:
            img1 = gaussians.sample_gaussians_img(
                self.means, self.conics, self.u[...,0], res, res, self.scale
            ).detach().cpu().numpy()

            img2 = gaussians.sample_gaussians_img(
                self.means, self.conics, self.u[...,1], res, res, self.scale
            ).detach().cpu().numpy()

            return np.stack([img1, img2])

    def plot_gaussians(self):
        return gaussians.plot_gaussians(self.means, self.covariances, self.u, self.scale)
