import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pysindy as ps
from pysindy.optimizers import FROLS
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plot

# ======================
# Generate Lorenz data
# ======================
dt = 0.01
T = 5
t = np.arange(0, T + dt, dt)

beta = 8/3
sigma = 10
rho = 28

def lorenz(x, t, sigma=sigma, beta=beta, rho=rho):
    x1, x2, x3 = x
    return [sigma * (x2 - x1),
            x1 * (rho - x3) - x2,
            x1 * x2 - beta * x3]

x0 = [-8, 8, 27]
x = odeint(lorenz, x0, t)

# ======================
# Feature library
# ======================
poly_lib = ps.PolynomialLibrary(degree=2)

# ======================
# Try different FROLS max_iter values
# ======================
iter_values = [3, 5, 7, 10, 15]
results = {}

for it in iter_values:
    print("\n" + "="*50)
    print(f"FROLS with max_iter = {it}")
    print("="*50)

    optimizer = FROLS(alpha=0.001, max_iter=it)
    model = ps.SINDy(feature_library=poly_lib, optimizer=optimizer)

    model.fit(x, t=dt)
    model.print()

    x_sim = model.simulate(x0, t)

    mse_total = mean_squared_error(x, x_sim)
    mse_vars = [mean_squared_error(x[:, i], x_sim[:, i]) for i in range(3)]

    print(f"Total MSE: {mse_total:.6f}")
    for lbl, mse in zip(['x', 'y', 'z'], mse_vars):
        print(f"MSE for {lbl}: {mse:.6f}")

    results[it] = {"model": model, "x_sim": x_sim, "mse_total": mse_total}

# ======================
# Find best iteration (least MSE)
# ======================
best_iter = min(results, key=lambda it: results[it]["mse_total"])
best_mse = results[best_iter]["mse_total"]

print("\n" + "="*60)
print(f"✅ Best FROLS max_iter = {best_iter}")
print(f"✅ Least Total MSE = {best_mse:.6f}")

# ======================
# Plot trajectories (time series)
# ======================
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
labels = ['x', 'y', 'z']

for i in range(3):
    axs[i].plot(t, x[:, i], 'k', label='Original')
    for it in iter_values:
        axs[i].plot(t, results[it]["x_sim"][:, i], '--', label=f'FROLS iter={it}')
    axs[i].set_ylabel(labels[i])
    axs[i].legend()

axs[2].set_xlabel("Time")
plt.tight_layout()
plt.show()

# ======================
# Bar plot: iteration vs total MSE
# ======================
mse_totals = [results[it]["mse_total"] for it in iter_values]

plt.figure(figsize=(8, 5))
plt.bar([str(it) for it in iter_values], mse_totals, color="skyblue", edgecolor="k")
plt.xlabel("FROLS max_iter")
plt.ylabel("Total MSE")
plt.title("Effect of FROLS Iterations on Total MSE (Lorenz system, SINDy)")
plt.show()

# ======================
# 3D Trajectory: Best Iter vs Actual
# ======================
x_sim_best = results[best_iter]["x_sim"]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot(x[:, 0], x[:, 1], x[:, 2], 'k', lw=2, label="Original Lorenz")
ax.plot(x_sim_best[:, 0], x_sim_best[:, 1], x_sim_best[:, 2],
        'r--', lw=2, label=f"SINDy (FROLS iter={best_iter})")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title(f"3D Trajectories: Original vs Best SINDy Model\nBest iter={best_iter}, MSE={best_mse:.6f}")
ax.legend()

plt.tight_layout()
plt.show()
