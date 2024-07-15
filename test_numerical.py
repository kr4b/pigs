import enum

import matplotlib.pyplot as plt
import numpy as np
import pde
import torch

class Problem(enum.Enum):
    DIFFUSION = enum.auto()
    BURGERS = enum.auto()
    WAVE = enum.auto()

scale = 2.5
res = 500
snapshots = 21
problem = Problem.BURGERS

samples = np.linspace(-1, 1, res) * scale
powers = -2.0 * samples ** 2
data = np.exp(powers).squeeze()

grid = pde.CartesianGrid([(-scale, scale)], [res])
state = pde.ScalarField(grid, data=data)
if problem == Problem.WAVE:
    state = pde.FieldCollection([state, state])

if problem == Problem.BURGERS:
    eq = pde.PDE({"u": "0.00318 * laplace(u) - u * d_dx(u)"})
    eq.check_implementation = False
    step = 1
    c = 1
elif problem == Problem.DIFFUSION:
    eq = pde.PDE({"u": "laplace(u)"})
    step = 1
    c = 1
elif problem == Problem.WAVE:
    eq = pde.PDE({"u": "10 * laplace(v) - 0.1 * u", "v": "u"})
    step = 1
    c = 2

results = []
for i in range(snapshots):
    result = eq.solve(state, dt=0.0001, t_range=0.05 * step * i)
    # result.plot()
    results.append(result.data)

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
        # plt.legend(["Prediction", "Ground Truth"])
    else:
        plt.plot(samples, prediction.sum(0))
        plt.plot(samples, results[i])

        ax = plt.gca()
        ax.set_xlim([-scale * 1.05, scale * 1.05])
        ax.set_ylim([-0.05, 1.05])
        # plt.legend(["Prediction", "Ground Truth"])

    for j in range(means.shape[0]):
        if problem == Problem.WAVE:
            # ax[0].plot(samples, prediction[j,:,0], "--")
            plt.plot(samples, prediction[j,:,1], "--")
        else:
            plt.plot(samples, prediction[j], "--")

    # plt.show()
    plt.savefig("../../notes/defense/explicit/frame_{}_{}.png".format(problem.name.lower(), i*step))

