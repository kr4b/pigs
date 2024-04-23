import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn

import gaussians

from diff_gaussian_sampling import GaussianSampler

nx = 20
ny = 20
d = 2

scale = 1.0

torch.manual_seed(0)

tx = torch.linspace(-1, 1, nx).cuda() * scale
ty = torch.linspace(-1, 1, ny).cuda() * scale
gx, gy = torch.meshgrid((tx,ty), indexing="ij")
means = torch.stack((gx,gy), dim=-1).reshape(nx*ny, d)
scaling = torch.exp(torch.ones((nx*ny,d), device="cuda") * -4.0)
transform = torch.zeros((nx*ny, d * (d - 1) // 2), device="cuda")

# samples = means.unsqueeze(-1) * 4
# powers = samples.transpose(-1, -2) @ samples
# values = torch.exp(-powers).squeeze(-1)
values = torch.zeros((nx*ny, 1), device="cuda")

for i in range(nx):
    if (i % 5) == 0:
        for j in range(ny):
            values[i * ny + j] = 1.0

covariances = gaussians.build_covariances(scaling, transform)
conics = torch.inverse(covariances)

# means = means.to(torch.float64)
# scaling = scaling.to(torch.float64)
# transform = transform.to(torch.float64)
# covariances = covariances.to(torch.float64)
# conics = conics.to(torch.float64)
# values = values.to(torch.float64)

gaussians.plot_gaussians(means, covariances, values)
plt.show()

res = 64
tx = torch.linspace(-1, 1, res).cuda() * scale
ty = torch.linspace(-1, 1, res).cuda() * scale
gx, gy = torch.meshgrid((tx, ty), indexing="xy")
samples = torch.stack((gx, gy), dim=-1).reshape(res * res, d)#.to(torch.float64)

sampler = GaussianSampler(True)
sampler.preprocess(means, values, covariances, conics, samples)
img = sampler.sample_gaussians().detach().cpu().numpy()

fig = plt.figure()
plt.imshow(img.reshape(res, res))
plt.axis("off")
plt.colorbar()
plt.show()

DISTANCE_EMBEDDINGS = 5
LATENT_SIZE = 16

activation = nn.ReLU()

input_projection = nn.Sequential(
    nn.Linear(1, LATENT_SIZE // 2),
    activation,
    nn.Linear(LATENT_SIZE // 2, LATENT_SIZE),
    activation,
    nn.Linear(LATENT_SIZE, LATENT_SIZE),
    # activation,
).cuda()

distance_transform = nn.Sequential(
    nn.Linear(LATENT_SIZE, (DISTANCE_EMBEDDINGS*d*LATENT_SIZE + LATENT_SIZE)//2),
    activation,
    nn.Linear((DISTANCE_EMBEDDINGS*d*LATENT_SIZE + LATENT_SIZE)//2,
               DISTANCE_EMBEDDINGS*d*LATENT_SIZE + LATENT_SIZE),
    # activation,
    # nn.Linear(DISTANCE_EMBEDDINGS*d*LATENT_SIZE + LATENT_SIZE,
    #            DISTANCE_EMBEDDINGS*d*LATENT_SIZE + LATENT_SIZE),
).cuda()

# features = torch.rand((nx*ny, LATENT_SIZE), device="cuda").to(torch.float64)
expected = torch.zeros((nx*ny, LATENT_SIZE), device="cuda")

for i in range(nx):
    if (i % 5) == 0:
        for j in range(ny):
            expected[(i + 1) * ny + j] = 1.0
            # features[i * ny + j] = 1.0

# distance_transforms = torch.zeros((nx*ny, LATENT_SIZE, d*DISTANCE_EMBEDDINGS + 1), device="cuda").to(torch.float64)
# distance_transforms.requires_grad = True
# features.requires_grad = True

# def test_func(features, distance_transforms):
#     indices, local_features = sampler.aggregate_neighbors(features, distance_transforms)
#     return local_features
# torch.autograd.gradcheck(test_func, (features, distance_transforms))
# print("Check")
# exit()

# features = nn.Parameter(features)
parameters = nn.ParameterList([
    *distance_transform.parameters(),
    *input_projection.parameters(),
    # features,
])
print(sum(p.numel() for p in parameters))

# optim = torch.optim.Adam(parameters, lr=1e-3)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)
# 
# start = time.time()
# for i in range(2000):
#     features = input_projection(values)
#     distance_transforms = distance_transform(features).reshape(nx*ny, LATENT_SIZE, d*DISTANCE_EMBEDDINGS + 1)
#     indices, local_features = sampler.aggregate_neighbors(features, distance_transforms)
#     distance_transforms = distance_transform(local_features).reshape(nx*ny, LATENT_SIZE, d*DISTANCE_EMBEDDINGS + 1)
#     indices, local_features = sampler.aggregate_neighbors(local_features, distance_transforms)
#     loss = torch.mean((local_features - expected) ** 2)
#     if (i % 100) == 0:
#         print(loss.item())
#     loss.backward()
#     optim.step()
#     optim.zero_grad()
#     # scheduler.step()
# 
# print(time.time() - start)
# exit()

features = torch.rand((nx*ny, LATENT_SIZE), device="cuda")
distance_transforms = torch.rand((nx*ny, LATENT_SIZE, d*DISTANCE_EMBEDDINGS + 1), device="cuda")
indices, local_features = sampler.aggregate_neighbors(features, distance_transforms)

for i in range(nx):
    print(features[i * ny])

for i in range(nx):
    print(local_features[i * ny])

index = (ny // 3 + 1) * nx + nx // 2 + 1
print("Overlaps:", torch.sum(indices[index]).item())
# print("Feature:", local_features[index])
# print("Feature:", local_features[index + 1])
gaussians.plot_gaussians(means[indices[index]], covariances[indices[index]], values[indices[index]])
plt.show()

sampler.preprocess(means[indices[index]], values[indices[index]], covariances[indices[index]], conics[indices[index]], samples)
img = sampler.sample_gaussians().detach().cpu().numpy()

fig = plt.figure()
plt.imshow(img.reshape(res, res))
plt.axis("off")
plt.colorbar()
plt.show()
