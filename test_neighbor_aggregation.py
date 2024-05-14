import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn

import gaussians

from diff_gaussian_sampling import GaussianSampler

nx = 25
ny = 25
d = 2

scale = 1.0

torch.manual_seed(0)

tx = torch.linspace(-1, 1, nx).cuda() * scale
ty = torch.linspace(-1, 1, ny).cuda() * scale
gx, gy = torch.meshgrid((tx,ty), indexing="ij")
means = torch.stack((gx,gy), dim=-1).reshape(nx*ny, d)
scaling = torch.exp(torch.ones((nx*ny,d), device="cuda") * -4.0)
transforms = torch.zeros((nx*ny, d * (d - 1) // 2), device="cuda")

# samples = means.unsqueeze(-1) * 4
# powers = samples.transpose(-1, -2) @ samples
# values = torch.exp(-powers).squeeze(-1)
values = torch.zeros((nx*ny, 1), device="cuda")

for i in range(nx):
    if (i % 5) == 0:
        for j in range(ny):
            values[i * ny + j] = 1.0

covariances, conics = gaussians.build_covariances(scaling, transforms)

gaussians.plot_gaussians(means, covariances, values)
# plt.show()

res = 64
tx = torch.linspace(-1, 1, res).cuda() * scale
ty = torch.linspace(-1, 1, res).cuda() * scale
gx, gy = torch.meshgrid((tx, ty), indexing="xy")
samples = torch.stack((gx, gy), dim=-1).reshape(res * res, d)

# means = means.to(torch.float64)
# scaling = scaling.to(torch.float64)
# transforms = transforms.to(torch.float64)
# covariances = covariances.to(torch.float64)
# conics = conics.to(torch.float64)
# values = values.to(torch.float64)
# samples = samples.to(torch.float64)

sampler = GaussianSampler(True)
sampler.preprocess(means, values, covariances, conics, samples)
img = sampler.sample_gaussians().detach().cpu().numpy()

fig = plt.figure()
plt.imshow(img.reshape(res, res))
plt.axis("off")
plt.colorbar()
# plt.show()

LATENT_SIZE = 8
KEY_SIZE = 4
EMBEDDING_SIZE = 21
FREQ_SIZE = (EMBEDDING_SIZE-1) // d // 2

# features = torch.rand((nx*ny, LATENT_SIZE), device="cuda").to(torch.float64)
# transforms = torch.rand((nx*ny, LATENT_SIZE), device="cuda").to(torch.float64)
# queries = torch.rand((nx*ny, KEY_SIZE), device="cuda").to(torch.float64)
# keys = torch.rand((nx*ny, KEY_SIZE), device="cuda").to(torch.float64)
# frequencies = torch.randn(FREQ_SIZE, device="cuda").to(torch.float64) * 10
# distance_transform = torch.rand((LATENT_SIZE, EMBEDDING_SIZE), device="cuda").to(torch.float64)
# 
# features.requires_grad = True
# transforms.requiers_grad = True
# queries.requiers_grad = True
# keys.requiers_grad = True
# frequencies.requires_grad = True
# distance_transform.requires_grad = True
# 
# def test_func(features, transforms, queries, keys, frequencies, distance_transform):
#     _, local_features = sampler.aggregate_neighbors(features, transforms, queries, keys, frequencies, distance_transform)
#     return local_features
# 
# # test_func(features, queries, frequencies, distance_transform)
# torch.autograd.gradcheck(test_func, (features, transforms, queries, keys, frequencies, distance_transform))
# print("Check")
# exit()

activation = nn.ReLU()

input_projection = nn.Sequential(
    nn.Linear(1, LATENT_SIZE // 2),
    activation,
    nn.Linear(LATENT_SIZE // 2, LATENT_SIZE),
    activation,
    nn.Linear(LATENT_SIZE, LATENT_SIZE),
    # activation,
).cuda()

# distance_transform_left = nn.Sequential(
#     nn.Linear(LATENT_SIZE, LATENT_SIZE),
#     activation,
#     nn.Linear(LATENT_SIZE, (LATENT_SIZE + LATENT_SIZE*KEY_SIZE)//2),
#     activation,
#     nn.Linear((LATENT_SIZE + LATENT_SIZE*KEY_SIZE)//2, LATENT_SIZE*KEY_SIZE),
#     # activation,
# ).cuda()
# 
# distance_transform_right = nn.Sequential(
#     nn.Linear(LATENT_SIZE, LATENT_SIZE),
#     activation,
#     nn.Linear(LATENT_SIZE, (LATENT_SIZE + KEY_SIZE*EMBEDDING_SIZE)//2),
#     activation,
#     nn.Linear((LATENT_SIZE + KEY_SIZE*EMBEDDING_SIZE)//2, KEY_SIZE*EMBEDDING_SIZE),
#     # activation,
# ).cuda()

distance_transform = torch.rand((LATENT_SIZE, EMBEDDING_SIZE), device="cuda")
transform = nn.Sequential(
    nn.Linear(LATENT_SIZE, LATENT_SIZE),
    activation,
    nn.Linear(LATENT_SIZE, LATENT_SIZE),
    activation,
    nn.Linear(LATENT_SIZE, LATENT_SIZE),
).cuda()
query_transform = nn.Sequential(
    nn.Linear(LATENT_SIZE, LATENT_SIZE),
    activation,
    nn.Linear(LATENT_SIZE, (LATENT_SIZE+KEY_SIZE)//2),
    activation,
    nn.Linear((LATENT_SIZE+KEY_SIZE)//2, KEY_SIZE),
).cuda()
key_transform = nn.Sequential(
    nn.Linear(LATENT_SIZE, LATENT_SIZE),
    activation,
    nn.Linear(LATENT_SIZE, (LATENT_SIZE+KEY_SIZE)//2),
    activation,
    nn.Linear((LATENT_SIZE+KEY_SIZE)//2, KEY_SIZE),
).cuda()
frequencies = torch.randn(FREQ_SIZE, device="cuda") * 10
expected = torch.zeros((nx*ny, LATENT_SIZE), device="cuda")

for i in range(nx):
    if (i % 5) == 0:
        for j in range(ny):
            expected[(i + 1) * ny + j] = 1.0

distance_transform = nn.Parameter(distance_transform)
frequencies = nn.Parameter(frequencies)
parameters = nn.ParameterList([
    distance_transform,
    *transform.parameters(),
    *query_transform.parameters(),
    *key_transform.parameters(),
    *input_projection.parameters(),
    frequencies,
])
print(sum(p.numel() for p in parameters))

optim = torch.optim.Adam(parameters, lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

start = time.time()
for i in range(2000):
    features = input_projection(values)
    transforms = transform(features)
    queries = query_transform(features)
    keys = key_transform(features)
    # dt_left = distance_transform_left(features).reshape(-1, LATENT_SIZE, KEY_SIZE)
    # dt_right = distance_transform_right(features).reshape(-1, KEY_SIZE, EMBEDDING_SIZE)
    # distance_transform = dt_left @ dt_right
    indices, local_features = sampler.aggregate_neighbors(
        features, transforms, queries, keys, frequencies, distance_transform)
    # indices, local_features = sampler.aggregate_neighbors(local_features, frequencies, distance_transform)
    loss = torch.mean((local_features - expected) ** 2)
    if (i % 100) == 0:
        print("Loss:", loss.item())
    loss.backward()
    optim.step()
    optim.zero_grad()
    # scheduler.step()

print(time.time() - start)
exit()

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
