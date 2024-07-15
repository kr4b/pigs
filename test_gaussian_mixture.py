import matplotlib.pyplot as plt
import numpy as np

means = np.array([-1.25, 0.0, 1.5]).reshape(1, 3)
variances = np.array([0.4, 0.75, 1.25]).reshape(1, 3)
values = np.array([0.85, 1.25, 0.4]).reshape(1, 3)

xs = np.linspace(-3.0, 5.0, 200)
samples = np.tile(xs, (means.shape[0], 1)).T - means
powers = -0.5 * samples * (1.0 / variances) * samples
ys = values * np.exp(powers)

plt.figure()
plt.plot(xs, ys[:,0], "--")
plt.plot(xs, ys[:,1], "--")
plt.plot(xs, ys[:,2], "--")
plt.plot(xs, np.sum(ys, -1), "black")
plt.legend(["N1", "N2", "N3", "Sum"])
plt.show()
