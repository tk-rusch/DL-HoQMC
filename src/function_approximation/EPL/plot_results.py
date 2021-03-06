import numpy as np
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()

Ns = (2 ** np.arange(7, 12)).tolist()
errors = np.zeros(len(Ns))
train_errors = np.zeros(len(Ns))

for i, N in enumerate(Ns):
    data = np.loadtxt('saved_results/N_' + str(int(N)) + '.txt')
    errors[i] = np.mean(data[:, 1])
    train_errors[i] = np.mean(data[:,0])

points = np.array(Ns) + np.array(Ns) / 2.

B = np.vstack([np.log(points), np.ones(len(Ns))]).T
m_gen, c_gen = np.linalg.lstsq(B, np.log(errors),rcond=None)[0]
print("decay rate generalization: ", -m_gen)

ax.loglog(points,errors,'blue',linestyle="",marker="o",label=r'$\mathcal{E}_G$ (rate: '+str(round(-m_gen,1))+')',basex=2,markersize=8,zorder=10)
ax.loglog(points, train_errors, 'blue', linestyle="", marker="s",label=r'$\mathcal{E}_T$', basex=2, markersize=8, zorder=10)

ax.loglog(points,np.exp(m_gen*np.log(points)+c_gen),'blue',linewidth=3)

plt.xlabel(r'\# sample points $N$')
plt.grid(b=True, which='both', linestyle='--')
plt.ylabel(r'Error')
plt.legend()
plt.title('Function approximation results in 50 dimensions')
plt.show()
