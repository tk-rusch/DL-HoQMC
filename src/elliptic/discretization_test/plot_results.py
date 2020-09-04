import numpy as np
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
Ns = (2 ** np.arange(3, 12)).tolist()
errors_1 = np.zeros(len(Ns))
errors_2 = np.zeros(len(Ns))
errors_3 = np.zeros(len(Ns))

for i, N in enumerate(Ns):
    data = np.loadtxt('saved_results_1/N_' + str(int(N)) + '.txt')
    errors_1[i] = np.abs(np.mean(data[:, 1]) - np.mean(data[:, 0]))

    data = np.loadtxt('saved_results_2/N_' + str(int(N)) + '.txt')
    errors_2[i] = np.abs(np.mean(data[:, 1]) - np.mean(data[:, 0]))

    data = np.loadtxt('saved_results_3/N_' + str(int(N)) + '.txt')
    errors_3[i] = np.abs(np.mean(data[:, 1]) - np.mean(data[:, 0]))

points = np.array(Ns) + np.array(Ns) / 2.

B = np.vstack([np.log(points), np.ones(len(Ns))]).T
m_gen1, c_gen1 = np.linalg.lstsq(B, np.log(errors_1),rcond=None)[0]
print("decay rate gap h_1: ", -m_gen1)

B = np.vstack([np.log(points), np.ones(len(Ns))]).T
m_gen2, c_gen2 = np.linalg.lstsq(B, np.log(errors_2),rcond=None)[0]
print("decay rate gap h_2: ", -m_gen2)

B = np.vstack([np.log(points), np.ones(len(Ns))]).T
m_gen3, c_gen3 = np.linalg.lstsq(B, np.log(errors_3),rcond=None)[0]
print("decay rate gap h_3: ", -m_gen3)

ax.loglog(points, errors_1, 'black', linestyle="", marker="o",
          label=r'$g_{h_1}$ (rate: ' + str(round(-m_gen1, 1)) + ')', basex=2, markersize=8, zorder=10)

ax.loglog(points, errors_2, 'blue', linestyle="", marker="s",
          label=r'$g_{h_2}$ (rate: ' + str(round(-m_gen2, 1)) + ')', basex=2, markersize=8, zorder=10)

ax.loglog(points, errors_3, 'grey', linestyle="", marker="^",
          label=r'$g_{h_3}$ (rate: ' + str(round(-m_gen3, 1)) + ')', basex=2, markersize=8, zorder=10)

ax.loglog(points, np.exp(m_gen1 * np.log(points) + c_gen1), 'black', linewidth=3)
ax.loglog(points, np.exp(m_gen2 * np.log(points) + c_gen2), 'blue', linewidth=3)
ax.loglog(points, np.exp(m_gen3 * np.log(points) + c_gen3), 'grey', linewidth=3)

plt.xlabel(r'\# sample points $N$')
plt.grid(b=True, which='both', linestyle='--')
plt.ylabel(r'$|\mathcal{E}_G - \mathcal{E}_T|$')
plt.legend()
plt.show()