import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn

import gaussians
import model

from model_pn import *

from diff_gaussian_sampling import GaussianSampler

train_timesteps = 10
cutoff_timesteps = 1
test_timesteps = 10

scale = 2.5

nx = ny = 30
d = 2

torch.manual_seed(1)
np.random.seed(1)

model = Model(
    Problem.BURGERS, 
    IntegrationRule.TRAPEZOID,
    nx, ny, d, scale
)

if model.problem == Problem.NAVIER_STOKES:
    gaussian_parameters = []
    for i in range(50):
        gaussian_parameters.append(torch.load("initialization/V1e-3/f_{}-small.pt".format(i)))

    file = np.load("ns_V1e-3_N50_T50.npy")
    f = torch.tensor(np.transpose(file, (3, 1, 2, 0)), device="cuda")

    means = gaussian_parameters[11]["means"].data
    values = gaussian_parameters[11]["values"].data
    scaling = gaussian_parameters[11]["scaling"].data
    transforms = gaussian_parameters[11]["transforms"].data

    model.set_initial_params(means, values, scaling, transforms)

    # for i in range(50):
    #     means = gaussian_parameters[i]["means"].data
    #     values = gaussian_parameters[i]["values"].data
    #     scaling = gaussian_parameters[i]["scaling"].data
    #     transforms = gaussian_parameters[i]["transforms"].data

    #     model.set_initial_params(means, values, scaling, transforms)
    #     model.plot_gaussians()
    #     plt.savefig("../training_data/V1e-3/gaussians_{}.png".format(i))

    #     res = 128
    #     tx = torch.linspace(-1, 1, res).cuda() * scale
    #     ty = torch.linspace(-1, 1, res).cuda() * scale
    #     gx, gy = torch.meshgrid((tx, ty), indexing="xy")
    #     samples = torch.stack((gx, gy), dim=-1).reshape(res * res, d)
    #     model.sampler.preprocess(model.means, model.u, model.covariances, model.conics, samples)

    #     img = model.sampler.sample_gaussians().reshape(res*res,2).detach().cpu().numpy()
    #     plt.figure()
    #     plt.imshow(img[:,0].reshape(res, res), cmap="plasma")
    #     plt.savefig("../training_data/V1e-3/frame_{}.png".format(i))

    # exit()

    # res = 128
    # tx = torch.linspace(-1, 1, res).cuda() * scale
    # ty = torch.linspace(-1, 1, res).cuda() * scale
    # gx, gy = torch.meshgrid((tx, ty), indexing="xy")
    # samples = torch.stack((gx, gy), dim=-1).reshape(res * res, d)
    # model.sampler.preprocess(model.means, model.u, model.covariances, model.conics, samples)

    # ux = model.sampler.sample_gaussians_derivative().reshape(res*res, d, 2)
    # img1 = ux[:,0,1] - ux[:,1,0]

    # desired = f[1,:,:,0]

    # res = 64
    # coords = ((samples + 1.0) / 2.0 * res).to(torch.long).clamp(0, res-1)
    # coords = coords[:,1] * res + coords[:,0]
    # img2 = torch.take(desired, coords).squeeze()

    # res = 128
    # fig = plt.figure()
    # ax = fig.subplots(1, 2)
    # im = ax[0].imshow(img1.reshape(res, res).detach().cpu().numpy())
    # plt.colorbar(im)
    # im = ax[1].imshow(img2.reshape(res, res).detach().cpu().numpy())
    # plt.colorbar(im)
    # plt.show()
    # plt.close(fig)

    # exit()

# for name, p in model.named_parameters():
#     print(name, p.numel())

print(sum(p.numel() for p in model.parameters()))

training_loss = []
mean_loss = []

log_step = 100
n_samples = 1024

optim = torch.optim.Adam(model.parameters())#, lr=1e-2)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.995)

dt = 1.0

start = 0

if len(sys.argv) > 1:
    state = torch.load(sys.argv[1])

    training_loss = state["training_loss"]
    # fig = plt.figure()
    # plt.plot(np.linspace(0, len(training_loss)*log_step, len(training_loss)), training_loss)
    # plt.yscale("log")
    # plt.show()
    # plt.close(fig)
    # exit()

    # prev_initial_u = model.initial_u.clone()
    # prev_initial_means = model.initial_means.clone()
    # prev_initial_scaling = model.initial_scaling.clone()
    # prev_initial_transforms = model.initial_transforms.clone()
    model.load_state_dict(state["model"])
    # model.initial_u = nn.Parameter(prev_initial_u)
    # model.initial_means = nn.Parameter(prev_initial_means)
    # model.initial_scaling = nn.Parameter(prev_initial_scaling)
    # model.initial_transforms = nn.Parameter(prev_initial_transforms)
    # model.set_initial_params(
    #     model.initial_means.detach().data, model.initial_u.detach().data, model.initial_scaling.detach().data, model.initial_transforms.detach().data)
    optim.load_state_dict(state["optimizer"])
    start = state["epoch"]
    training_loss = state["training_loss"]

gaussian_parameters = torch.load("initialization/init_gaussian2.pt")
means = gaussian_parameters["means"] * scale
values = gaussian_parameters["values"]
scaling = gaussian_parameters["scaling"] * scale * scale
transforms = gaussian_parameters["transforms"]

model.set_initial_params(means, values, scaling, transforms)

# model.reset(False)
# model.randomize(nx)
# 
# fig = model.plot_gaussians()
# # plt.savefig("../../notes/training-figures/init_uniform_gaussians.pdf")
# plt.show()
# plt.close(fig)
# 
# res = 128
# img = model.generate_images(res)
# 
# if model.problem == Problem.WAVE:
#     for i in range(2):
#         plt.figure()
#         plt.imshow(img[i])
#         plt.colorbar()
#         plt.show()
# else:
#     plt.figure()
#     plt.imshow(img[0])
#     plt.colorbar()
#     plt.show()
#     # plt.axis("off")
#     # plt.savefig("../../notes/training-figures/init_uniform.png", bbox_inches="tight")
# 
# exit()

# tx = torch.linspace(-1, 1, res).cuda() * scale
# ty = torch.flip(torch.linspace(-1, 1, res).cuda().unsqueeze(-1), (0,1)).squeeze() * scale
# gx, gy = torch.meshgrid((tx, ty), indexing="xy")
# samples = torch.stack((gx, gy), dim=-1).reshape(res * res, d)
# 
# ones = torch.ones((model.means[model.boundary_mask.squeeze()].shape[0], 1), device="cuda")
# model.sampler.preprocess(model.means[model.boundary_mask.squeeze()], ones, model.covariances[model.boundary_mask.squeeze()], model.conics[model.boundary_mask.squeeze()], samples)
# density = model.sampler.sample_gaussians().reshape(res, res, 1).detach().cpu().numpy()
# 
# plt.figure()
# plt.imshow(density)
# plt.colorbar()
# plt.show()
# 
# exit()

if len(sys.argv) <= 1 or "--resume" in sys.argv:
    os.makedirs("checkpoints", exist_ok=True)

    model.train()

    torch.autograd.set_detect_anomaly(True)

    n_samples = 1024
    N = 5000
    log_step = 10
    save_step = 100
    bootstrap_rate = 100
    split_epoch = 10000
    epsilon = 1

    current_timesteps = 1
    epoch = start

    for epoch in range(start, N):
        time_samples = torch.rand(n_samples, device="cuda")
        samples = (torch.rand((n_samples, d), device="cuda") * 2.0 - 1.0) * scale

        boundaries = torch.cat((
            -torch.ones(n_samples//4, device="cuda") - torch.rand(n_samples//4, device="cuda") * 0.5,
            torch.ones(n_samples//4, device="cuda") + torch.rand(n_samples//4, device="cuda") * 0.5
        )) * scale
        # if model.problem == Problem.NAVIER_STOKES:
        #     bc_samples = \
        #         torch.zeros((n_samples + n_samples // 4, d), device="cuda")
        # else:
        #     bc_samples = torch.zeros((n_samples, d), device="cuda")
        bc_samples = torch.zeros((n_samples, d), device="cuda")

        bc_samples[n_samples // 2:n_samples,0] = \
            (torch.rand(n_samples // 2, device="cuda") * 2.0 - 1.0) * 1.5 * scale
        bc_samples[n_samples // 2:n_samples,1] = boundaries
        bc_samples[:n_samples // 2,1] = \
            (torch.rand(n_samples // 2, device="cuda") * 2.0 - 1.0) * 1.5 * scale
        bc_samples[:n_samples // 2,0] = boundaries

        # if model.problem == Problem.NAVIER_STOKES:
        #     hypersphere = torch.rand((n_samples // 4, 1), device="cuda") * 2.0 - 1.0
        #     for i in range(d - 1):
        #         r = 1.0 - (hypersphere ** 2).sum(-1).reshape(-1, 1)
        #         hypersphere = torch.cat((
        #             hypersphere,
        #             (torch.rand((n_samples // 4, 1), device="cuda") * 2.0 - 1.0) * r
        #         ), dim=-1)

        #     bc_samples[n_samples:,:] = \
        #         (hypersphere * 0.1 - torch.tensor([[0.65, 0.0]], device="cuda")) * scale

        total_loss = 0
        total_pde_loss = 0
        total_bc_loss = 0
        total_conservation_loss = 0
        total_initial_loss = 0
        total_magnitude_loss = 0

        if model.problem == Problem.NAVIER_STOKES:
            data_index = np.random.randint(len(gaussian_parameters))

            means = gaussian_parameters[data_index]["means"].data.detach().clone()
            values = gaussian_parameters[data_index]["values"].data.detach().clone()
            scaling = gaussian_parameters[data_index]["scaling"].data.detach().clone()
            transforms = gaussian_parameters[data_index]["transforms"].data.detach().clone()
            model.set_initial_params(means, values, scaling, transforms)

            total_recon_loss = 0
        # elif model.problem != Problem.TEST:
        else:
            model.randomize(np.random.randint(15, 40))
            model.reset(False) # epoch <= split_epoch)

            if np.random.rand() > 0.5:
                gaussian_parameters = torch.load("initialization/double_gaussian2.pt")
                means = gaussian_parameters["means"].data * scale
                values = gaussian_parameters["values"].data
                scaling = gaussian_parameters["scaling"].data * scale * scale
                transforms = gaussian_parameters["transforms"].data

                model.set_initial_params(means, values, scaling, transforms)
            else:
                gaussian_parameters = torch.load("initialization/init_gaussian2.pt")
                means = gaussian_parameters["means"].data * scale
                values = gaussian_parameters["values"].data
                scaling = gaussian_parameters["scaling"].data * scale * scale
                transforms = gaussian_parameters["transforms"].data

                model.set_initial_params(means, values, scaling, transforms)

        model.sample(samples, bc_samples)

        loss_weight = 1
        loss = torch.zeros(1, device="cuda")

        all_sufficient = True

        res = 64
        tx = torch.linspace(-1, 1, res).cuda() * scale
        ty = torch.linspace(-1, 1, res).cuda() * scale
        gx, gy = torch.meshgrid((tx, ty), indexing="xy")
        img_samples = torch.stack((gx, gy), dim=-1).reshape(res * res, d)

        for g in optim.param_groups:
            g["prev_lr"] = g["lr"]

        for i in range(min(min(epoch // bootstrap_rate + 1, current_timesteps), train_timesteps)):
            pde_loss = torch.zeros(1, device="cuda")
            bc_loss = torch.zeros(1, device="cuda")
            conservation_loss = torch.zeros(1, device="cuda")
            initial_loss = torch.zeros(1, device="cuda")
            magnitude_loss = torch.zeros(1, device="cuda")

            t = i * dt

            model.forward(t, dt, epoch > split_epoch)
            losses = model.compute_loss(t, dt, samples, time_samples, bc_samples)

            if not torch.isnan(losses[0]) and not torch.isinf(losses[0]):
                pde_loss += losses[0]
            if not torch.isnan(losses[1]) and not torch.isinf(losses[1]):
                bc_loss += losses[1]
            if not torch.isnan(losses[2]) and not torch.isinf(losses[2]):
                conservation_loss += losses[2]
            if not torch.isnan(losses[3]) and not torch.isinf(losses[3]):
                initial_loss += losses[3]
            if not torch.isnan(losses[4]) and not torch.isinf(losses[4]):
                magnitude_loss += losses[4]

            total_pde_loss += pde_loss.item()
            total_bc_loss += bc_loss.item()
            total_conservation_loss += conservation_loss.item()
            total_initial_loss += initial_loss.item()
            total_magnitude_loss += magnitude_loss.item()

            # if epoch < 20:
            #     current_loss = conservation_loss
            # else:
            current_loss = pde_loss + bc_loss + conservation_loss + initial_loss

            if model.problem == Problem.NAVIER_STOKES:
                desired = f[data_index,:,:,i+1]

                res = 64
                coords = ((samples + 1.0) / 2.0 * res).to(torch.long).clamp(0, res-1)
                coords = coords[:,1] * res + coords[:,0]

                recon_loss = 5.0 * torch.mean(
                    (model.w_samples[-1].squeeze() - torch.take(desired, coords).squeeze()) ** 2)
                total_recon_loss += recon_loss.item()
                current_loss += recon_loss

            # current_loss += 0.1 * current_loss.item() * magnitude_loss
            loss = current_loss

            for g in optim.param_groups:
                g["lr"] = g["prev_lr"] * loss_weight

            loss.backward()
            optim.step()
            optim.zero_grad()

            print(i, current_loss.item(), loss_weight)
            loss_weight *= np.exp(-epsilon * current_loss.item())

            total_loss += current_loss.item()
            all_sufficient &= current_loss < 1.0

            model.clear()
            model.sample(samples, bc_samples)
            model.detach()

        for g in optim.param_groups:
            g["lr"] = g["prev_lr"]

        if all_sufficient:
            current_timesteps = min(epoch // bootstrap_rate + 1, current_timesteps) + 1

        # if loss > 0:
        #     loss.backward()
        #     optim.step()
        #     optim.zero_grad()
        # scheduler.step()

        if (epoch+1) % log_step == 0:
            training_loss.append(total_loss / (i+1) * train_timesteps)
            print("Epoch {}: Total Loss {}".format(epoch, training_loss[-1]))
            print("  BC Loss:", total_bc_loss)
            print("  PDE Loss:", total_pde_loss)
            print("  Conservation Loss:", total_conservation_loss)
            # print("  Initial Loss:", total_initial_loss)
            print("  Magnitude Loss:", total_magnitude_loss)
            if model.problem == Problem.NAVIER_STOKES:
                print("  Reconstruction Loss:", total_recon_loss)
                print("  Data index:", data_index)

        if (epoch+1) % save_step == 0:
            torch.save({
                "epoch": epoch + 2,
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "training_loss": training_loss,
            }, "results_model_pn/{}_model_{}.pt".format(model.problem.name.lower(), epoch))

            fig = plt.figure()
            plt.plot(np.linspace(0, len(training_loss)*log_step, len(training_loss)), training_loss)
            plt.yscale("log")
            plt.savefig("results_model_pn/training_loss.png")
            plt.close(fig)

    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "training_loss": training_loss,
    }, "results_model_pn/{}_model.pt".format(model.problem.name.lower()))

torch.manual_seed(0)

# model.reset(False)

imgs = []
vmin = np.inf
vmax = -np.inf

total_loss = 0.0

gt = np.load("burgers_simple_gt.npy")

total_time = 0
evo_time = 0
total_norm = 0

with torch.no_grad():
    res = 64
    tx = torch.linspace(-1, 1, res).cuda() * scale
    ty = torch.flip(torch.linspace(-1, 1, res).cuda().unsqueeze(-1), (0,1)).squeeze() * scale
    gx, gy = torch.meshgrid((tx, ty), indexing="xy")
    samples = torch.stack((gx, gy), dim=-1).reshape(res * res, d)

    prev_u = np.zeros((res*res, model.channels, 1))
    prev_ux = np.zeros((res*res, model.channels * d, 1))
    prev_uxx = np.zeros((res*res, model.channels * d*d, 1))
    if model.problem == Problem.NAVIER_STOKES:
        prev_w = np.zeros((res*res))
        prev_uxxx = np.zeros((res*res, model.channels*d*d*d, 1))

    for i in range(test_timesteps):
        fig = model.plot_gaussians()
        # plt.axis("off")
        # plt.savefig("../../notes/results/burgers/simple/gaussians{}.png".format(i), bbox_inches="tight")
        plt.close(fig)

        mask = model.boundary_mask.squeeze()
        model.sampler.preprocess(
            model.means[mask], model.u[mask], model.covariances[mask], model.conics[mask], samples)

        nt = 10
        time_samples = (np.arange(1, nt+1) / nt).reshape(1, 1, -1).repeat(res*res, axis=0)

        start = time.time()
        u = model.sampler.sample_gaussians().reshape(res*res, -1, 1).detach().cpu().numpy()
        if i == test_timesteps - 1:
            evo_time += time.time() - start
        total_time += time.time() - start

        ux = \
           model.sampler.sample_gaussians_derivative().reshape(res*res, d, -1).detach().cpu().numpy()
        uxx = \
           model.sampler.sample_gaussians_laplacian().reshape(res*res,-1,1).detach().cpu().numpy()
        if model.problem == Problem.NAVIER_STOKES:
            uxxx = model.sampler.sample_gaussians_third_derivative() \
                        .reshape(res*res, -1, 1).detach().cpu().numpy()
        sample_u = time_samples * u.reshape(res*res, -1, 1) \
                 - (1.0 - time_samples) * prev_u.reshape(res*res, -1, 1)
        sample_ux = time_samples * ux.reshape(res*res, -1, 1) \
                  - (1.0 - time_samples) * prev_ux.reshape(res*res, -1, 1)
        sample_uxx = time_samples * uxx.reshape(res*res, -1, 1) \
                   - (1.0 - time_samples) * prev_uxx.reshape(res*res, -1, 1)
        if model.problem == Problem.NAVIER_STOKES:
            sample_uxxx = time_samples * uxxx.reshape(res*res, -1, 1) \
                        - (1.0 - time_samples) * prev_uxxx.reshape(res*res, -1, 1)

        sample_ux = sample_ux.reshape(res*res, d, model.channels, -1)
        sample_uxx = sample_uxx.reshape(res*res, d, d, model.channels, -1)
        if model.problem == Problem.NAVIER_STOKES:
            sample_uxxx = sample_uxxx.reshape(res*res, d, d, d, model.channels, -1)
            w = ux[:,0,1] - ux[:,1,0]
            wx = sample_uxx[:,:,0,1] - sample_uxx[:,:,1,0]
            wxx = sample_uxxx[:,:,:,0,1] -  sample_uxxx[:,:,:,1,0]

        if model.problem == Problem.NAVIER_STOKES:
            loss = np.zeros((res*res))
            sample_pde = np.zeros((res*res))
        else:
            loss = np.zeros((res*res, model.channels))
            sample_pde = np.zeros((res*res, model.channels))

        for j in range(nt):
            if model.problem == Problem.NAVIER_STOKES:
                rhs = model.pde_rhs(
                    samples,sample_u[...,j],sample_ux[...,j],sample_uxx[...,j],wx[...,j],wxx[...,j])
                loss += (w - prev_w) - dt * rhs
            else:
                rhs = model.pde_rhs(samples, sample_u[...,j], sample_ux[...,j], sample_uxx[...,j])
                loss += (u - prev_u).squeeze(-1) - dt * rhs

            sample_pde += rhs

        sample_pde = dt * sample_pde.reshape(res, res, -1) / nt
        loss = loss.reshape(res, res, -1) / nt
        if i > 0:
            total_loss += (loss ** 2).sum().item()
            print((loss ** 2).sum().item())

        # laplacian = uxx[...,0,0].reshape(res, res, 1) + uxx[...,-1,0].reshape(res, res, 1)
        # ut = u.reshape(res, res, 1) - prev_u.reshape(res, res, 1)

        if model.problem == Problem.NAVIER_STOKES:
            imgs.append(w.reshape(res, res))
        else:
            imgs.append(u.reshape(res, res, -1))

        if i > 0:
            desired = torch.tensor(gt[i], device="cuda")

            im_res = 500
            coords = ((samples / scale + 1.0) / 2.0 * im_res).to(torch.long).clamp(0, im_res-1)
            coords = coords[:,1] * im_res + coords[:,0]

            desired = torch.take(desired.T, coords).reshape(res, res)
            norm = torch.norm(torch.tensor(imgs[-1], device="cuda").squeeze() - desired) / torch.norm(desired)
            total_norm += norm.item()

        # fig = plt.figure()
        # ax = fig.subplots(2, 2)
        # im = ax[0,0].imshow(ut)
        # ax[0,0].set_title("u_t")
        # plt.colorbar(im)
        # im = ax[0,1].imshow(laplacian)
        # ax[0,1].set_title("Laplacian")
        # plt.colorbar(im)
        # im = ax[1,0].imshow(sample_pde)
        # ax[1,0].set_title("PDE")
        # plt.colorbar(im)
        # im = ax[1,1].imshow(loss ** 2)
        # ax[1,1].set_title("Loss")
        # plt.colorbar(im)
        # plt.tight_layout()
        # plt.savefig("results_model_pn/loss_avg_{}.png".format(i))
        # plt.close(fig)

        # ones = torch.ones((model.means[model.boundary_mask.squeeze()].shape[0], 1), device="cuda")
        # model.sampler.preprocess(model.means[model.boundary_mask.squeeze()], ones, model.covariances[model.boundary_mask.squeeze()], model.conics[model.boundary_mask.squeeze()], samples)
        # density = model.sampler.sample_gaussians().reshape(res, res, 1).detach().cpu().numpy()
        # density = 1.0 - (density - density.min()) / density.max()

        # fig = plt.figure()
        # ax = fig.subplots(2, 2)
        # im = ax[0,0].imshow(density)
        # ax[0,0].set_title("Density")
        # plt.colorbar(im)
        # im = ax[0,1].imshow(density * ut ** 2)
        # ax[0,1].set_title("* u_t")
        # plt.colorbar(im)
        # im = ax[1,0].imshow(density * laplacian ** 2)
        # ax[1,0].set_title("* Laplacian")
        # plt.colorbar(im)
        # im = ax[1,1].imshow(density * loss ** 2)
        # ax[1,1].set_title("* Loss")
        # plt.colorbar(im)
        # plt.tight_layout()
        # plt.savefig("results_model_pn/density_{}.png".format(i))
        # plt.close(fig)

        vmin = min(vmin, np.min(imgs[-1]))
        vmax = max(vmax, np.max(imgs[-1]))

        t = i * dt
        start = time.time()
        model.forward(t, dt, False)
        evo_time += time.time() - start
        total_time += time.time() - start
        # plt.savefig("results_model_pn/split{}.png".format(i))

        prev_u = u
        prev_ux = ux
        prev_uxx = uxx
        if model.problem == Problem.NAVIER_STOKES:
            prev_w = w
            prev_uxxx = uxxx

    for i in range(test_timesteps):
        fig = plt.figure()

        if model.problem == Problem.WAVE:
            ax = fig.subplots(1, 2)
            im = ax[0].imshow(imgs[i][...,0], vmin=vmin, vmax=vmax)
            plt.colorbar(im)
            im = ax[1].imshow(imgs[i][...,1], vmin=vmin, vmax=vmax)
            plt.colorbar(im)

        else:
            plt.imshow(imgs[i], vmin=vmin, vmax=vmax)#, cmap="plasma")
            plt.colorbar()
            # plt.colorbar()

        # plt.axis("off")
        plt.savefig("../../notes/results/burgers/simple/frame{}.png".format(i), bbox_inches="tight")
        # plt.close(fig)

print("Time (full):", total_time)
print("Time (evo):", evo_time)
print("Loss:", total_loss)
print("Norm:", total_norm / test_timesteps)
