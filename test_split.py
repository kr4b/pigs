import matplotlib.pyplot as plt
import numpy as np
import torch

import gaussians

means = np.array([-1.25, 0.0, 1.5]).reshape(1, 3)
variances = np.array([0.4, 0.75, 1.25]).reshape(1, 3)
values = np.array([0.85, 1.25, 0.4]).reshape(1, 3)

i1, i2 = 0, -1
means[0,i1], means[0,i2] = means[0,i2], means[0,i1]
variances[0,i1], variances[0,i2] = variances[0,i2], variances[0,i1]
values[0,i1], values[0,i2] = values[0,i2], values[0,i1]

xs = np.linspace(-3.0, 5.0, 200)
samples = np.tile(xs, (means.shape[0], 1)).T - means
powers = -0.5 * samples * (1.0 / variances) * samples
ys = values * np.exp(powers)

sum_before = np.sum(ys, -1)

plt.figure()
plt.plot(xs, ys[:,0], "--")
plt.plot(xs, ys[:,1], "--")
plt.plot(xs, ys[:,2], "--")
plt.plot(xs, sum_before, "black")
plt.legend(["N1", "N2", "N3", "Sum"])
plt.savefig("before_split.pdf")

std = np.sqrt(variances[0,-1]) / 4.0
means = np.concatenate((means[0,:2], np.array([means[0,-1] - std, means[0,-1] + std])))
variances = np.concatenate((variances[0,:2], np.repeat(variances[0,-1], 2)))
values = np.concatenate((values[0,:2], np.repeat(values[0,-1] / 2.0, 2)))

samples = np.tile(xs, (means.shape[0], 1)).T - means
powers = -0.5 * samples * (1.0 / variances) * samples
ys = values * np.exp(powers)

sum_after = np.sum(ys, -1)

plt.figure()
plt.plot(xs, ys[:,0], "--")
plt.plot(xs, ys[:,1], "--")
plt.plot(xs, ys[:,2], "--")
plt.plot(xs, ys[:,3], "--")
plt.plot(xs, sum_after, "black")
plt.plot(xs, sum_before, color="gray", linestyle="--")
plt.legend(["N1", "N2", "N3_1", "N3_2", "Sum", "Original"])
plt.savefig("after_split.pdf")

fig = plt.figure()
ax = fig.subplots(1, 2)
ax[0].plot(xs, sum_before)
ax[0].plot(xs, sum_after)
ax[1].plot(xs, sum_before - sum_after)
plt.savefig("split_compare.png")

print(np.sum((sum_before - sum_after) ** 2))
