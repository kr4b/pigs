import enum

import matplotlib.pyplot as plt
import numpy as np
import pde
import torch

class Problem(enum.Enum):
    DIFFUSION = enum.auto()
    BURGERS = enum.auto()
    WAVE = enum.auto()

d = 2
scale = 2.5
res = 500
snapshots = 200
problem = Problem.BURGERS

storage = pde.storage.MemoryStorage()

sample_mean = np.array([0.0, 0.0]).reshape(1, d, 1) * scale
sample_mean2 = np.array([0.0, -0.6]).reshape(1, d, 1) * scale

tx = np.linspace(-1, 1, res) * scale
ty = np.linspace(-1, 1, res) * scale
gx, gy = np.meshgrid(tx, ty)
samples = np.stack((gx, gy), axis=-1).reshape(res * res, d, 1)
samples_ = samples - sample_mean
conics = np.linalg.inv(np.eye(d) * 0.05 * scale * scale).reshape(1, d, d).repeat(res*res, 0)
powers = -0.5 * np.matmul(samples_.transpose(0, 2, 1), np.matmul(conics, samples_))
data = np.exp(powers).squeeze().reshape(res, res)# * 0.5
# samples_ = samples - sample_mean2
# conics = np.linalg.inv(np.diag([0.1, 0.025]) * scale * scale)
# powers = -0.5 * (samples_.transpose(0, 2, 1) @ (conics @ samples_))
# data += np.exp(powers).squeeze().reshape(res, res)

grid = pde.CartesianGrid([(-scale, scale), (-scale, scale)], [res, res])
state = pde.ScalarField(grid, data=data)

if problem == Problem.WAVE:
    state = pde.FieldCollection([state, state])

if problem == Problem.BURGERS:
    bc_x = {"value": 0}
    bc_y = {"value": 0}
    eq = pde.PDE({"u": "0.0318 * laplace(u) - u * d_dx(u)"}, bc=[bc_x, bc_y])
    eq.check_implementation = False
    c = 1
elif problem == Problem.DIFFUSION:
    eq = pde.PDE({"u": "laplace(u)"}, bc={"value": 0})
    c = 1
elif problem == Problem.WAVE:
    eq = pde.PDE({"u": "10 * laplace(v) - 0.1 * u", "v": "u"})
    c = 2

vmin = 0.0
vmax = 1.0
dt = 0.0001
T = 1.0
eq.solve(state, dt=dt, t_range=T, tracker=storage.tracker(T/snapshots))

results = []
for i in range(snapshots):
    result = storage[i].data
    results.append(result)
    vmin = min(vmin, result.min())
    vmax = max(vmax, result.max())

for i in range(snapshots):
    fig = plt.figure()
    plt.imshow(results[i].T, vmin=vmin, vmax=vmax, cmap="viridis")
    plt.axis("off")
    plt.savefig("../../notes/defense/ground_truth/burgers/frame_{}.png".format(i), bbox_inches="tight")
    plt.close(fig)

# plt.figure()
# plt.imshow([[0]], vmin=1.01, vmax=-0.01, cmap="viridis")
# plt.colorbar()
# plt.axis("off")
# plt.savefig("../../notes/defense/diffusion/empty.png", bbox_inches="tight")

# np.save("burgers_simple_gt.npy", np.array(results))

exit()

for i in range(snapshots):
    gaussians = torch.load(
        "results_no_mlp_1d/gaussians_{}_{}.pt".format(problem.name.lower(), i * step),
        map_location=torch.device("cpu"))

    means = np.tanh(gaussians["means"].detach().numpy()).reshape(-1, 1, 1) * 2.5
    values = gaussians["values"].detach().numpy().reshape(-1, 1, c)
    covariances = np.exp(gaussians["scaling"].detach().numpy()).reshape(-1, 1, 1)
    conics = 1.0 / covariances
    prediction = (values * np.exp(-0.5 * conics * (samples.reshape(1, -1, 1) - means) ** 2))

    fig = plt.figure()
    if problem == Problem.WAVE:
        # ax = fig.subplots(1, 2)
        # ax[0].plot(samples, prediction.sum(0)[:,0])
        plt.plot(samples, prediction.sum(0)[:,1])
        # ax[0].plot(samples, results[i][1])
        plt.plot(samples, results[i][0])

        # ax[0].set_xlim([-scale * 1.05, scale * 1.05])
        # ax[0].set_ylim([-0.05, 1.05])
        # ax[1].set_xlim([-scale * 1.05, scale * 1.05])
        # ax[1].set_ylim([-0.05, 1.05])

        # ax[0].legend(["Prediction", "Ground Truth"])
        plt.legend(["Prediction", "Ground Truth"])
    else:
        plt.plot(samples, prediction.sum(0))
        plt.plot(samples, results[i])

        ax = plt.gca()
        ax.set_xlim([-scale * 1.05, scale * 1.05])
        ax.set_ylim([-0.05, 1.05])
        plt.legend(["Prediction", "Ground Truth"])

    for j in range(means.shape[0]):
        if problem == Problem.WAVE:
            # ax[0].plot(samples, prediction[j,:,0], "--")
            plt.plot(samples, prediction[j,:,1], "--")
        else:
            plt.plot(samples, prediction[j], "--")

    # plt.show()
    plt.savefig("../../notes/explicit-figures/frame_{}_{}.pdf".format(problem.name.lower(), i*step))

