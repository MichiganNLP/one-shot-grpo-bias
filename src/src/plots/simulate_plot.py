import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

def solve_z(a, b, t):
    def equation(z):
        return 2*b*z + np.exp(a+b*z) - np.exp(-(a+b*z)) - np.exp(a) + np.exp(-a) - b*t
    # Provide a reasonable bracket for root finding
    # Try to guess an interval around zero
    z_min, z_max = -10, 10
    try:
        sol = root_scalar(equation, bracket=[z_min, z_max], method='brentq')
    except:
        return 0
    if not sol.converged:
        return 0
        raise RuntimeError("Root finding did not converge")

    return sol.root

def logistic_regression():
    x = np.array([1.])
    eta = 0.1
    ts = np.arange(1, 3001)
    theta0_range = np.arange(-10, 1, 1)  # from -10 to 0 inclusive

    plt.figure(figsize=(8, 4))

    # Use a colormap for smooth gradient colors
    import matplotlib as mpl
    cmap = plt.get_cmap('viridis')
    num_curves = len(theta0_range)
    norm = mpl.colors.Normalize(vmin=0, vmax=num_curves-1)

    for idx, theta0_val in enumerate(theta0_range):
        theta0 = np.array([theta0_val])
        a = np.sum(x*theta0)
        b = np.sum(x**2)

        thetas = []
        for t in ts:
            z_val = solve_z(a, b, t*eta)
            theta = theta0 + z_val * x 
            thetas.append(theta)

        ppos = [100/(1+np.exp(-np.sum(thetas[i]*x))) for i in range(len(thetas))]
        color = cmap(norm(idx))
        plt.plot(ts, ppos, color=color, label=rf'Margin: {np.abs(theta0_val)}')

    plt.xlabel('Step', fontsize=15)
    plt.ylabel(r'Training Accuracy %', fontsize=15)
    plt.ylim(0, 100)
    plt.legend(loc='lower right', fontsize='medium', ncol=2)
    plt.tight_layout()
    plt.savefig('../figures/simulation/training_acc_vs_t.pdf')
    plt.close()

if __name__ == "__main__":
    logistic_regression()