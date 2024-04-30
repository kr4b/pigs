import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn

import gaussians

from diff_gaussian_sampling import GaussianSampler

torch.manual_seed(0)

nx = 10
ny = 10
d = 2

scale = 1.0

tx = torch.linspace(-1, 1, nx).cuda() * scale
ty = torch.linspace(-1, 1, ny).cuda() * scale
gx, gy = torch.meshgrid((tx,ty), indexing="ij")
means = torch.stack((gx,gy), dim=-1).reshape(nx, ny, d)
scaling = torch.ones((nx,ny,d), device="cuda") * -3.5
transform = torch.zeros((nx,ny, d * (d - 1) // 2), device="cuda")
for i in range(ny):
    for j in range(nx):
        transform[j,i,0] = 0.5

samples = means.unsqueeze(-1) * 4
powers = samples.transpose(-1, -2) @ samples
values = torch.exp(-powers).squeeze(-1)
# values = values + torch.rand(values.shape, device="cuda") * 0.5
# values = torch.zeros((nx,ny), device="cuda")
# 
# for i in range(nx):
#     for j in range(ny):
#         x = i / (nx//2) - 1.0
#         y = j / (ny//2) - 1.0
#         if y > 0.2 and 0.3 < x**2 + y**2 < 0.6:
#             values[i,j] = 1.0
# 
# values[nx//3,ny//3] = 3.0
# values[nx//3*2,ny//3] = 3.0
# values[nx//3-1,ny//3] = 2.0
# values[nx//3-2,ny//3] = 3.0
# values[nx//3*2+1,ny//3] = 2.0
# values[nx//3*2+2,ny//3] = 3.0
# scaling[nx//3,ny//3] = -3.5
# scaling[nx//3-2,ny//3] = -3.5
# scaling[nx//3*2,ny//3] = -3.5
# scaling[nx//3*2+2,ny//3] = -3.5

scaling = torch.exp(scaling)

full_covariances, full_conics = gaussians.build_full_covariances(scaling, transform)
full_covariances = full_covariances.reshape(-1, d, d)
full_conics = full_conics.reshape(-1, d, d)
covariances, conics = gaussians.build_covariances(scaling, transform)

res = 64
tx = torch.linspace(-1, 1, res).cuda() * scale
ty = torch.linspace(-1, 1, res).cuda() * scale
gx, gy = torch.meshgrid((tx, ty), indexing="xy")
samples = torch.stack((gx, gy), dim=-1).reshape(res * res, d)

means = means.reshape(-1, d)#.to(torch.float64)
values = values.reshape(-1, 1)#.to(torch.float64)
covariances = covariances.reshape(-1, d * (d + 1) // 2)#.to(torch.float64)
conics = conics.reshape(-1, d * (d + 1) // 2)#.to(torch.float64)
samples = samples.reshape(-1, d)#.to(torch.float64)

gaussians.plot_gaussians(means, covariances, values)
plt.savefig("gaussian_plot.png")

means.requires_grad = True
values.requires_grad = True
conics.requires_grad = True
samples.requires_grad = True
full_conics.requires_grad = True

sampler = GaussianSampler(True)
sampler.preprocess(means, values, covariances, conics, samples)

# def test_sample(*args):
#     return sampler.sample_gaussians()
# 
# def test_derivative(*args):
#     return sampler.sample_gaussians_derivative()
# 
# def test_laplacian(*args):
#     return sampler.sample_gaussians_laplacian()
# 
# def test_third(*args):
#     return sampler.sample_gaussians_third_derivative()
# 
# torch.autograd.gradcheck(test_sample, (means, values, conics))
# print("Check sample")
# 
# torch.autograd.gradcheck(test_derivative, (means, values, conics))
# print("Check derivative")
# 
# torch.autograd.gradcheck(test_laplacian, (means, values, conics))
# print("Check laplacian")
# 
# torch.autograd.gradcheck(test_third, (means, values, conics))
# print("Check third")
# 
# exit()

result_py = gaussians.sample_gaussians(means, full_conics, values, samples)
result_cuda = sampler.sample_gaussians()

input_img = result_py.reshape(res, res).detach().cpu().numpy()
# input_img = (-2*samples.reshape(res,res,2) * (samples.unsqueeze(-1).transpose(-2, -1) @ samples.unsqueeze(-1)).reshape(res,res,1)).detach().cpu().numpy()

fig = plt.figure()
plt.imshow(input_img)
plt.axis("off")
plt.colorbar()
plt.savefig("derivative_input.png")

grad_x_auto = torch.autograd.grad(result_py.sum(), (means, values, full_conics, samples), retain_graph=True, create_graph=True)
grad_x_cuda_auto = torch.autograd.grad(result_cuda.sum(), (means, values, conics), retain_graph=True, create_graph=True)
grad_x_cuda = sampler.sample_gaussians_derivative().squeeze()

# for i in range(len(grad_x_cuda_auto)):
#     assert(torch.allclose(grad_x_auto[i], grad_x_cuda_auto[i], atol=10e-4))
# 
# assert(torch.allclose(grad_x_cuda_auto[-1], grad_x_cuda, atol=10e-4))

h = scale * 2 / (res - 1)
img_fd_x = (input_img[1:-1,1:-1] - input_img[1:-1,:-2]) / h
img_fd_y = (input_img[1:-1,1:-1] - input_img[:-2,1:-1]) / h
img_fd = np.stack((img_fd_x, img_fd_y), axis=2)

grad_x_auto = grad_x_auto[-1]

img_py = gaussians.gaussian_derivative(means, full_conics, values, samples)
img_py = img_py.reshape(res, res, -1).detach().cpu().numpy()

img_auto = grad_x_auto.reshape(res, res, -1).detach().cpu().numpy()
img_cuda = grad_x_cuda.reshape(res, res, -1).detach().cpu().numpy()
error_py_auto = img_py - img_auto
error_py_cuda = img_py - img_cuda

fig = plt.figure()
ax = fig.subplots(1, 2)
ax[0].set_title("x")
im = ax[0].imshow(img_py[:,:,0])
plt.colorbar(im)
ax[1].set_title("y")
im = ax[1].imshow(img_py[:,:,1])
plt.colorbar(im)
plt.savefig("derivatives_py.png")

fig = plt.figure()
ax = fig.subplots(1, 2)
ax[0].set_title("x")
im = ax[0].imshow(img_fd[:,:,0])
plt.colorbar(im)
ax[1].set_title("y")
im = ax[1].imshow(img_fd[:,:,1])
plt.colorbar(im)
plt.savefig("derivatives_fd.png")

fig = plt.figure()
ax = fig.subplots(1, 2)
ax[0].set_title("x")
im = ax[0].imshow(img_auto[:,:,0])
plt.colorbar(im)
ax[1].set_title("y")
im = ax[1].imshow(img_auto[:,:,1])
plt.colorbar(im)
plt.savefig("derivatives_autograd.png")

fig = plt.figure()
ax = fig.subplots(1, 2)
ax[0].set_title("x")
im = ax[0].imshow(img_cuda[:,:,0])
plt.colorbar(im)
ax[1].set_title("y")
im = ax[1].imshow(img_cuda[:,:,1])
plt.colorbar(im)
plt.savefig("derivatives_cuda.png")

fig = plt.figure()
ax = fig.subplots(1, 2)
ax[0].set_title("x")
im = ax[0].imshow(error_py_auto[:,:,0])
plt.colorbar(im)
ax[1].set_title("y")
im = ax[1].imshow(error_py_auto[:,:,1])
plt.colorbar(im)
plt.savefig("derivatives_error_py_auto.png")

fig = plt.figure()
ax = fig.subplots(1, 2)
ax[0].set_title("x")
im = ax[0].imshow(error_py_cuda[:,:,0])
plt.colorbar(im)
ax[1].set_title("y")
im = ax[1].imshow(error_py_cuda[:,:,1])
plt.colorbar(im)
plt.savefig("derivatives_error_py_cuda.png")

plt.close("all")

grad_xx_auto = torch.autograd.grad(grad_x_auto[...,0].sum(), (means, values, full_conics, samples), retain_graph=True, create_graph=True)
grad_yy_auto = torch.autograd.grad(grad_x_auto[...,1].sum(), (means, values, full_conics, samples), retain_graph=True, create_graph=True)
lap_auto = []
for i in range(len(grad_xx_auto)):
    lap_auto.append(torch.cat((grad_xx_auto[i].unsqueeze(-1), grad_yy_auto[i].unsqueeze(-1)), dim=-1))

grad_xx_cuda_auto = torch.autograd.grad(grad_x_cuda[...,0].sum(), (means, values, conics), retain_graph=True)
grad_yy_cuda_auto = torch.autograd.grad(grad_x_cuda[...,1].sum(), (means, values, conics))
lap_cuda_auto = []
for i in range(len(grad_xx_cuda_auto)):
    lap_cuda_auto.append(torch.cat((grad_xx_cuda_auto[i].unsqueeze(-1), grad_yy_cuda_auto[i].unsqueeze(-1)), dim=-1))

lap_cuda = sampler.sample_gaussians_laplacian().squeeze()

# assert(torch.allclose(lap_cuda_auto[-1].reshape(lap_cuda.shape), lap_cuda, atol=10e-3))
# 
# for i in range(len(lap_cuda_auto)):
#     assert(torch.allclose(lap_auto[i], lap_cuda_auto[i].reshape(lap_auto[i].shape), atol=10e-3))

lap_auto = lap_auto[-1]

img_fd_xx = -(2 * input_img[1:-1,1:-1] - input_img[1:-1,:-2] - input_img[1:-1,2:]) / (h**2)
img_fd_yy = -(2 * input_img[1:-1,1:-1] - input_img[2:,1:-1] - input_img[:-2,1:-1]) / (h**2)
img_fd_xy = (input_img[:-2,:-2] + input_img[2:,2:] - input_img[2:,:-2] - input_img[:-2,2:]) / (4*h**2)
img_fd_yx = img_fd_xy
img_fd = np.stack((img_fd_xx, img_fd_xy, img_fd_yx, img_fd_yy), axis=2)

img_py = gaussians.gaussian_derivative2(means, full_conics, values, samples)
img_py = img_py.reshape(res, res, -1).detach().cpu().numpy()
img_auto = lap_auto.reshape(res, res, -1).detach().cpu().numpy()
img_cuda = lap_cuda.reshape(res, res, -1).detach().cpu().numpy()
error_py_auto = img_py - img_auto
error_py_cuda = img_py - img_cuda

fig = plt.figure()
ax = fig.subplots(2, 2)
ax[0,0].set_title("xx")
im = ax[0,0].imshow(img_py[:,:,0])
plt.colorbar(im)
ax[0,1].set_title("xy")
im = ax[0,1].imshow(img_py[:,:,1])
plt.colorbar(im)
ax[1,0].set_title("yx")
im = ax[1,0].imshow(img_py[:,:,2])
plt.colorbar(im)
ax[1,1].set_title("yy")
im = ax[1,1].imshow(img_py[:,:,3])
plt.colorbar(im)
plt.savefig("second_derivatives_py.png")

fig = plt.figure()
ax = fig.subplots(2, 2)
ax[0,0].set_title("xx")
im = ax[0,0].imshow(img_fd[:,:,0])
plt.colorbar(im)
ax[0,1].set_title("xy")
im = ax[0,1].imshow(img_fd[:,:,1])
plt.colorbar(im)
ax[1,0].set_title("yx")
im = ax[1,0].imshow(img_fd[:,:,2])
plt.colorbar(im)
ax[1,1].set_title("yy")
im = ax[1,1].imshow(img_fd[:,:,3])
plt.colorbar(im)
plt.savefig("second_derivatives_fd.png")

fig = plt.figure()
ax = fig.subplots(2, 2)
ax[0,0].set_title("xx")
im = ax[0,0].imshow(img_auto[:,:,0])
plt.colorbar(im)
ax[0,1].set_title("xy")
im = ax[0,1].imshow(img_auto[:,:,1])
plt.colorbar(im)
ax[1,0].set_title("yx")
im = ax[1,0].imshow(img_auto[:,:,2])
plt.colorbar(im)
ax[1,1].set_title("yy")
im = ax[1,1].imshow(img_auto[:,:,3])
plt.colorbar(im)
plt.savefig("second_derivatives_autograd.png")

fig = plt.figure()
ax = fig.subplots(2, 2)
ax[0,0].set_title("xx")
im = ax[0,0].imshow(img_cuda[:,:,0])
plt.colorbar(im)
ax[0,1].set_title("xy")
im = ax[0,1].imshow(img_cuda[:,:,1])
plt.colorbar(im)
ax[1,0].set_title("yx")
im = ax[1,0].imshow(img_cuda[:,:,2])
plt.colorbar(im)
ax[1,1].set_title("yy")
im = ax[1,1].imshow(img_cuda[:,:,3])
plt.colorbar(im)
plt.savefig("second_derivatives_cuda.png")

fig = plt.figure()
ax = fig.subplots(2, 2)
ax[0,0].set_title("xx")
im = ax[0,0].imshow(error_py_auto[:,:,0])
plt.colorbar(im)
ax[0,1].set_title("xy")
im = ax[0,1].imshow(error_py_auto[:,:,1])
plt.colorbar(im)
ax[1,0].set_title("yx")
im = ax[1,0].imshow(error_py_auto[:,:,2])
plt.colorbar(im)
ax[1,1].set_title("yy")
im = ax[1,1].imshow(error_py_auto[:,:,3])
plt.colorbar(im)
plt.savefig("second_derivatives_error_py_auto.png")

fig = plt.figure()
ax = fig.subplots(2, 2)
ax[0,0].set_title("xx")
im = ax[0,0].imshow(error_py_cuda[:,:,0])
plt.colorbar(im)
ax[0,1].set_title("xy")
im = ax[0,1].imshow(error_py_cuda[:,:,1])
plt.colorbar(im)
ax[1,0].set_title("yx")
im = ax[1,0].imshow(error_py_cuda[:,:,2])
plt.colorbar(im)
ax[1,1].set_title("yy")
im = ax[1,1].imshow(error_py_cuda[:,:,3])
plt.colorbar(im)
plt.savefig("second_derivatives_error_py_cuda.png")

plt.close("all")

grad_xx_auto = torch.autograd.grad(lap_auto[...,0,0].sum(), (means, values, full_conics, samples), retain_graph=True)
grad_xy_auto = torch.autograd.grad(lap_auto[...,0,1].sum(), (means, values, full_conics, samples), retain_graph=True)
grad_yx_auto = torch.autograd.grad(lap_auto[...,1,0].sum(), (means, values, full_conics, samples), retain_graph=True)
grad_yy_auto = torch.autograd.grad(lap_auto[...,1,1].sum(), (means, values, full_conics, samples))

grad_auto = []
for i in range(len(grad_xx_auto)):
    grad_auto.append(torch.cat((grad_xx_auto[i].unsqueeze(-1), grad_xy_auto[i].unsqueeze(-1), grad_yx_auto[i].unsqueeze(-1), grad_yy_auto[i].unsqueeze(-1)), dim=-1))

grad_xx_cuda_auto = torch.autograd.grad(lap_cuda[...,0,0].sum(), (means, values, conics), retain_graph=True)
grad_xy_cuda_auto = torch.autograd.grad(lap_cuda[...,0,1].sum(), (means, values, conics), retain_graph=True)
grad_yx_cuda_auto = torch.autograd.grad(lap_cuda[...,1,0].sum(), (means, values, conics), retain_graph=True)
grad_yy_cuda_auto = torch.autograd.grad(lap_cuda[...,1,1].sum(), (means, values, conics))

grad_cuda_auto = []
for i in range(len(grad_xx_cuda_auto)):
    grad_cuda_auto.append(torch.cat((grad_xx_cuda_auto[i].unsqueeze(-1), grad_xy_cuda_auto[i].unsqueeze(-1), grad_yx_cuda_auto[i].unsqueeze(-1), grad_yy_cuda_auto[i].unsqueeze(-1)), dim=-1))

# for i in range(len(grad_cuda_auto)):
#     assert(torch.allclose(grad_auto[i], grad_cuda_auto[i].reshape(grad_auto[i].shape), atol=10e-4))
