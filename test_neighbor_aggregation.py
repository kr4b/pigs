import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn

import gaussians

from diff_gaussian_sampling import GaussianSampler

gradcheck = True

nx = ny = 5 if gradcheck else 25
d = 2

scale = 1.0

torch.manual_seed(0)

tx = torch.linspace(-1, 1, nx).cuda() * scale
ty = torch.linspace(-1, 1, ny).cuda() * scale
gx, gy = torch.meshgrid((tx,ty), indexing="ij")
means = torch.stack((gx,gy), dim=-1).reshape(nx*ny, d)
scaling = torch.exp(torch.ones((nx*ny,d), device="cuda") * (-1.5 if gradcheck else -4.0))
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

if gradcheck:
    means = means.to(torch.float64)
    scaling = scaling.to(torch.float64)
    transforms = transforms.to(torch.float64)
    covariances = covariances.to(torch.float64)
    conics = conics.to(torch.float64)
    values = values.to(torch.float64)
    samples = samples.to(torch.float64)

sampler = GaussianSampler(True)
sampler.preprocess(means, values, covariances, conics, samples)
img = sampler.sample_gaussians().detach().cpu().numpy()

fig = plt.figure()
plt.imshow(img.reshape(res, res))
plt.axis("off")
plt.colorbar()
# plt.show()

LATENT_SIZE = 2 if gradcheck else 16
KEY_SIZE = 4
EMBEDDING_SIZE = 21
FREQ_SIZE = (EMBEDDING_SIZE-1) // d // 2

if gradcheck:
    features = torch.rand((nx*ny, LATENT_SIZE), device="cuda").to(torch.float64)
    transform = torch.rand((LATENT_SIZE, LATENT_SIZE), device="cuda").to(torch.float64)
    queries = torch.rand((nx*ny, KEY_SIZE), device="cuda").to(torch.float64)
    keys = torch.rand((nx*ny, KEY_SIZE), device="cuda").to(torch.float64)
    frequencies = torch.randn(FREQ_SIZE, device="cuda").to(torch.float64) * 10
    distance_transform = torch.rand((LATENT_SIZE, EMBEDDING_SIZE*2), device="cuda").to(torch.float64)

    features.requires_grad = True
    transform.requires_grad = True
    queries.requires_grad = True
    keys.requires_grad = True
    frequencies.requires_grad = True
    distance_transform.requires_grad = True

    def test_func(features, transform, queries, keys, frequencies, distance_transform):
        local_features = sampler.aggregate_neighbors(
            features, transform, queries, keys, frequencies, distance_transform)
        return local_features

    start = time.time()

    sampler.preprocess_aggregate()
    torch.autograd.gradcheck(
        test_func, (features, transform, queries, keys, frequencies, distance_transform))

    print("Check:", time.time() - start)
    exit()

activation = nn.ReLU()

heads = 2

norms = [nn.LayerNorm(LATENT_SIZE).cuda() for _ in range(heads)]

input_projection = nn.Sequential(
    nn.Linear(1, LATENT_SIZE // 2),
    activation,
    nn.Linear(LATENT_SIZE // 2, LATENT_SIZE),
    activation,
    nn.Linear(LATENT_SIZE, LATENT_SIZE),
    # activation,
).cuda()
output_projection = nn.Sequential(
    nn.Linear(heads * LATENT_SIZE, LATENT_SIZE),
    activation,
    nn.Linear(LATENT_SIZE, LATENT_SIZE // 2),
    activation,
    nn.Linear(LATENT_SIZE // 2, 1),
    # activation,
).cuda()

distance_transform = torch.rand((LATENT_SIZE, EMBEDDING_SIZE*2), device="cuda")
transform = torch.rand((heads, LATENT_SIZE, LATENT_SIZE), device="cuda")
query_transform = torch.rand((heads, KEY_SIZE, LATENT_SIZE), device="cuda")
key_transform = torch.rand((heads, KEY_SIZE, LATENT_SIZE), device="cuda")
frequencies = torch.randn(FREQ_SIZE, device="cuda") * 10
expected = torch.zeros((nx*ny, 1), device="cuda")

for i in range(nx):
    if (i % 5) == 0:
        for j in range(ny):
            expected[(i + 1) * ny + j] = 1.0

transform = nn.Parameter(transform)
distance_transform = nn.Parameter(distance_transform)
frequencies = nn.Parameter(frequencies)
parameters = nn.ParameterList([
    distance_transform,
    transform,
    query_transform,
    key_transform,
    *input_projection.parameters(),
    *output_projection.parameters(),
    frequencies,
])
print(sum(p.numel() for p in parameters))

optim = torch.optim.Adam(parameters, lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

sampler.preprocess(means, values, covariances, conics, means)

start = time.time()
for i in range(2000):
    features = input_projection(values)
    all_features = None
    sampler.preprocess_aggregate()

    magnitudes = torch.zeros(heads, device="cuda")

    for j in range(heads):
        queries = (query_transform[j] @ features.unsqueeze(-1)).squeeze(-1)
        keys = (key_transform[j] @ features.unsqueeze(-1)).squeeze(-1)
        local_features = sampler.aggregate_neighbors(
            features, transform[j], queries, keys, frequencies, distance_transform)
        magnitudes[j] = (local_features ** 2).mean()
        local_features = norms[j](local_features)
        if all_features == None:
            all_features = local_features
        else:
            all_features = torch.cat((all_features, local_features), dim=-1)

    # for j in range(heads):
    #     all_features[...,j*LATENT_SIZE:(j+1)*LATENT_SIZE] /= torch.sqrt(magnitudes[j])

    output = output_projection(all_features)
    loss = torch.mean((output - expected) ** 2)
    magnitude_loss = ((magnitudes - magnitudes.mean()) ** 2).sum()
    # loss += loss.item() * magnitude_loss
    if (i % 100) == 0:
        print(magnitudes)
        print("Magnitude loss:", magnitude_loss.item())
        print("Loss:", loss.item())
    loss.backward()
    optim.step()
    optim.zero_grad()
    # scheduler.step()

print(time.time() - start)
exit()

features = torch.rand((nx*ny, LATENT_SIZE), device="cuda")
distance_transform = torch.rand((nx*ny, LATENT_SIZE, d*DISTANCE_EMBEDDINGS + 1), device="cuda")
local_features = sampler.aggregate_neighbors(features, distance_transform)

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
