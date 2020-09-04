import numpy as np
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(8, 6))
ax = fig.gca()

Ns = (2**np.arange(3,12)).tolist()
errors = np.zeros(len(Ns))
errors2 = np.zeros(len(Ns))

for i, N in enumerate(Ns):
    data = np.loadtxt('saved_results16/N_'+str(int(N))+'.txt')
    errors[i] = np.abs(np.mean(data[:,1]) - np.mean(data[:,0]))

    data2 = np.loadtxt('saved_results32/N_' + str(int(N)) + '.txt')
    errors2[i] = np.abs(np.mean(data2[:,1]) - np.mean(data2[:,0]))

points = np.array(Ns) + np.array(Ns) / 2.

B = np.vstack([np.log(points), np.ones(len(Ns))]).T
m_gen, c_gen = np.linalg.lstsq(B, np.log(errors),rcond=None)[0]
print("decay rate gap 16 dim: ",-m_gen)

B2 = np.vstack([np.log(points), np.ones(len(Ns))]).T
m_gen2, c_gen2 = np.linalg.lstsq(B2, np.log(errors2),rcond=None)[0]
print("decay rate gap 32 dim: ",-m_gen2)

plt.loglog(points,errors,'blue',linestyle="",marker="o",label=r'$d=16$ (rate: '+str(round(-m_gen,1))+')',basex=2,markersize=8,zorder=10)
plt.loglog(points,errors2,'black',linestyle="",marker="s",label=r'$d=32$ (rate: '+str(round(-m_gen2,1))+')',basex=2,markersize=8,zorder=8)
ax.loglog(points,np.exp(m_gen*np.log(points)+c_gen),'blue',linewidth=3)
ax.loglog(points,np.exp(m_gen2*np.log(points)+c_gen2),'black',linewidth=3)

plt.tick_params(axis="y", labelsize=18)
plt.tick_params(axis="x", labelsize=18)
plt.xlabel(r'\# sample points $N$')
plt.grid(b=True, which='both', linestyle='--')
plt.legend()
plt.ylabel(r'$|\mathcal{E}_G - \mathcal{E}_T|$')
plt.show()