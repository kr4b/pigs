import time
import torch
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

nx = 100
ny = 100

cov = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]])

tx = torch.linspace(-1, 1, nx)
ty = torch.linspace(-1, 1, ny)
gx, gy = torch.meshgrid((tx,ty), indexing="ij")
gz = torch.ones((nx,ny))
means = torch.stack((gx,gy,gz), dim=-1)
values = torch.tensor(st.multivariate_normal.pdf(means, mean=[0,0,0], cov=cov), dtype=torch.float32)
values = values / torch.max(values)

means = means.unsqueeze(-1).reshape(-1, 3, 1)
# x = means[:,:,0]
# y = means[:,:,1]

con = torch.inverse(cov)

# This is the 2D multivariate normal without normalization e^(-0.5 * x^T C^-1 x)
# It should work for any dimension
power = -0.5 * (means.transpose(-1, -2) @ (con @ means))
samples = torch.exp(power.squeeze()).reshape(nx, ny)

fig = plt.figure()
ax = fig.subplots(1, 3)

im = ax[0].imshow(values)
plt.colorbar(im)

im = ax[1].imshow(samples)
plt.colorbar(im)

im = ax[2].imshow(torch.abs(samples - values))
plt.colorbar(im)

plt.show()
