import numpy as np
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()

Ns = (2 ** np.arange(3, 12)).tolist()
errorsl1 = np.zeros(len(Ns))
errorsl3 = np.zeros(len(Ns))

for i, N in enumerate(Ns):
    data = np.loadtxt('saved_results_L1/N_' + str(int(N)) + '.txt')
    errorsl1[i] = np.abs(np.mean(data[:, 1]) - np.mean(data[:, 0]))

    data = np.loadtxt('saved_results_L3/N_' + str(int(N)) + '.txt')
    errorsl3[i] = np.abs(np.mean(data[:, 1]) - np.mean(data[:, 0]))

points = np.array(Ns) + np.array(Ns) / 2.

B = np.vstack([np.log(points), np.ones(len(Ns))]).T
m_gen1, c_gen1 = np.linalg.lstsq(B, np.log(errorsl1),rcond=None)[0]
print("decay rate gap L1: ", -m_gen1)

m_gen3, c_gen3 = np.linalg.lstsq(B, np.log(errorsl3),rcond=None)[0]
print("decay rate gap L3: ", -m_gen3)

ax.loglog(points, errorsl1, 'blue', linestyle="", marker="o", label=r'$L^1$ (rate: ' + str(round(-m_gen1, 1)) + ')',
          basex=2, markersize=8, zorder=10)
ax.loglog(points, errorsl3, 'black', linestyle="", marker="s", label=r'$L^3$ (rate: ' + str(round(-m_gen3, 1)) + ')',
          basex=2, markersize=8, zorder=10)


ax.loglog(points, np.exp(m_gen1 * np.log(points) + c_gen1), 'blue', linewidth=3)
ax.loglog(points, np.exp(m_gen3 * np.log(points) + c_gen3), 'black', linewidth=3)

plt.xlabel(r'\# sample points $N$')
plt.grid(b=True, which='both', linestyle='--')
plt.ylabel(r'$|\mathcal{E}_G - \mathcal{E}_T|$')
plt.legend()
plt.show()
