import numpy as np
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()
Ns = (2 ** np.arange(7, 13)).tolist()
Ns2 = (2 ** np.arange(6, 13)).tolist()

gap = np.zeros(len(Ns))
gap_extra = np.zeros(len(Ns2))

for i, N in enumerate(Ns):
    data = np.loadtxt('saved_results/N_' + str(int(N)) + '.txt')
    gap[i] = np.abs(np.mean(data[:, 1]) - np.mean(data[:, 0]))

for i, N in enumerate(Ns2):
    data = np.loadtxt('../EPL/saved_results/N_' + str(int(N)) + '.txt')
    gap_extra[i] = np.abs(np.mean(data[:, 1]) - np.mean(data[:, 0]))

points = np.array(Ns2) + np.array(Ns2) / 2.

B = np.vstack([np.log(np.array(Ns)), np.ones(len(Ns))]).T
m_gen, c_gen = np.linalg.lstsq(B, np.log(gap),rcond=None)[0]
print("decay rate gap IPL: ", -m_gen)

B2 = np.vstack([np.log(points), np.ones(len(points))]).T
m_gen2, c_gen2 = np.linalg.lstsq(B2, np.log(gap_extra),rcond=None)[0]
print("decay rate gap EPL: ", -m_gen2)

ax.loglog(points, gap_extra, 'blue', linestyle="", marker="o", label=r'EPL (rate: ' + str(round(-m_gen2, 1)) + ')',
          basex=2, markersize=8, zorder=10)
ax.loglog(Ns, gap, 'black', linestyle="", marker="s", label=r'IPL (rate: ' + str(round(-m_gen, 1)) + ')', basex=2,
          markersize=8, zorder=10)

ax.loglog(points, np.exp(m_gen2 * np.log(points) + c_gen2), 'blue', linewidth=3)
ax.loglog(points, np.exp(m_gen * np.log(points) + c_gen), 'black', linewidth=3)

plt.xlabel(r'\# sample points $N$')
plt.grid(b=True, which='both', linestyle='--')
plt.ylabel(r'$|\mathcal{E}_G - \mathcal{E}_T|$')
plt.legend()
plt.title('IPL vs EPL function approximation experiment in 50d')
plt.show()
