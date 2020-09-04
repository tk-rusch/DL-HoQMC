import numpy as np
from matplotlib import pyplot as plt

def plot_relu_vs_tanh():
    plt.rc('text', usetex=True)
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    Ns = (2 ** np.arange(4, 9)).tolist()
    errors_rel = np.zeros(len(Ns))
    train_errors_rel = np.zeros(len(Ns))

    errors_tanh = np.zeros(len(Ns))
    train_errors_tanh = np.zeros(len(Ns))

    for i, N in enumerate(Ns):
        data = np.loadtxt('saved_results_relu/N_' + str(int(N)) + '.txt')
        errors_rel[i] = np.mean(data[:, 1])
        train_errors_rel[i] = np.mean(data[:,0])

        data = np.loadtxt('../dimensionality_test/saved_results16/N_' + str(int(N)) + '.txt')
        errors_tanh[i] = np.mean(data[:, 1])
        train_errors_tanh[i] = np.mean(data[:, 0])

    points = np.array(Ns) + np.array(Ns) / 2.

    B = np.vstack([np.log(points), np.ones(len(Ns))]).T
    m_gen2, c_gen2 = np.linalg.lstsq(B, np.log(errors_rel),rcond=None)[0]
    print("decay rate generalization relu: ", -m_gen2)

    B = np.vstack([np.log(points), np.ones(len(Ns))]).T
    m_gen, c_gen = np.linalg.lstsq(B, np.log(errors_tanh),rcond=None)[0]
    print("decay rate generalization tanh: ", -m_gen)

    ax.loglog(points, errors_tanh, 'black', linestyle="", marker="o",
              label=r'$\mathcal{E}_G$ Tanh (rate: ' + str(round(-m_gen, 1)) + ')', basex=2, markersize=8, zorder=10)
    ax.loglog(points, train_errors_tanh, 'black', linestyle="", marker="s", label=r'$\mathcal{E}_T$ Tanh', basex=2,
              markersize=8, zorder=10)

    ax.loglog(points, errors_rel, 'blue', linestyle="", marker="o",
              label=r'$\mathcal{E}_G$ ReLU (rate: ' + str(round(-m_gen2, 1)) + ')', basex=2, markersize=8, zorder=10)
    ax.loglog(points, train_errors_rel, 'blue', linestyle="", marker="s", label=r'$\mathcal{E}_T$ ReLU', basex=2,
              markersize=8, zorder=10)

    ax.loglog(points,np.exp(m_gen*np.log(points)+c_gen),'black',linewidth=3)
    ax.loglog(points, np.exp(m_gen2 * np.log(points) + c_gen2), 'blue', linewidth=3)

    plt.xlabel(r'\# sample points $N$')
    plt.grid(b=True, which='both', linestyle='--')
    plt.ylabel(r'Error')
    plt.legend()
    plt.show()

def plot_relu_vs_tanh_gap():
    plt.rc('text', usetex=True)
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    Ns = (2 ** np.arange(3, 11)).tolist()
    errors_rel = np.zeros(len(Ns))
    errors_tanh = np.zeros(len(Ns))

    for i, N in enumerate(Ns):
        data = np.loadtxt('saved_results_relu/N_' + str(int(N)) + '.txt')
        errors_rel[i] = np.abs(np.mean(data[:, 1]) - np.mean(data[:,0]))

        data = np.loadtxt('../dimensionality_test/saved_results16/N_' + str(int(N)) + '.txt')
        errors_tanh[i] = np.abs(np.mean(data[:, 1]) - np.mean(data[:,0]))

    points = np.array(Ns) + np.array(Ns) / 2.

    B = np.vstack([np.log(points), np.ones(len(Ns))]).T
    m_gen2, c_gen2 = np.linalg.lstsq(B, np.log(errors_rel),rcond=None)[0]
    print("decay rate gap relu: ", -m_gen2)

    B = np.vstack([np.log(points), np.ones(len(Ns))]).T
    m_gen, c_gen = np.linalg.lstsq(B, np.log(errors_tanh),rcond=None)[0]
    print("decay rate gap tanh: ", -m_gen)

    ax.loglog(points, errors_tanh, 'black', linestyle="", marker="o",
              label=r'Tanh (rate: ' + str(round(-m_gen, 1)) + ')', basex=2, markersize=8, zorder=10)

    ax.loglog(points, errors_rel, 'blue', linestyle="", marker="o",
              label=r'ReLU (rate: ' + str(round(-m_gen2, 1)) + ')', basex=2, markersize=8, zorder=10)

    ax.loglog(points,np.exp(m_gen*np.log(points)+c_gen),'black',linewidth=3)
    ax.loglog(points, np.exp(m_gen2 * np.log(points) + c_gen2), 'blue', linewidth=3)

    plt.xlabel(r'\# sample points $N$')
    plt.grid(b=True, which='both', linestyle='--')
    plt.ylabel(r'$|\mathcal{E}_G - \mathcal{E}_T|$')
    plt.legend()
    plt.show()

plot_relu_vs_tanh()
plot_relu_vs_tanh_gap()
