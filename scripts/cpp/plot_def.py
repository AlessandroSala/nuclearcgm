
import numpy as np
import matplotlib.pyplot as plt

def roundHI(x):
    ceil = np.ceil(x*2)
    floor = np.floor(x*2)
    ceil[ceil % 2 == 0] = floor[ceil % 2 == 0]
    return ceil / 2

def momFromSq(x):
    return 0.5 * (-1 + np.sqrt(1 + 4 * x))

# Caricamento dati
data = np.genfromtxt('output/def_energies.csv')
ms = roundHI(np.genfromtxt('output/m.csv'))
L2 = np.round((np.genfromtxt('output/L2.csv')))
J2 = roundHI((np.genfromtxt('output/J2.csv')))
P = np.round(np.genfromtxt('output/P.csv'))

data = data[::2]
ms = ms[::2]
L2 = L2[::2]
J2 = J2[::2]
P = P[::2]

n_rows, n_cols = data.shape
assignments = np.zeros((n_rows, n_cols))
scores = np.zeros((n_rows, n_cols))

# Definizione delle shell
shells = [
    {"l": 0, "j": 0.5, "m": [0.5, -0.5], "P": 1},
    {"l": 1, "j": 1.5, "m": [1.5, -1.5], "P": -1},
    {"l": 1, "j": 1.5, "m": [0.5, -0.5], "P": -1},
    {"l": 1, "j": 0.5, "m": [0.5, -0.5], "P": -1},
    {"l": 2, "j": 2.5, "m": [0.5, -0.5], "P": 1},
    {"l": 2, "j": 2.5, "m": [1.5, -1.5], "P": 1},
    {"l": 2, "j": 2.5, "m": [2.5, -2.5], "P": 1},
    {"l": 0, "j": 0.5, "m": [0.5, -0.5], "P": 1},
    {"l": 2, "j": 1.5, "m": [0.5, -0.5], "P": 1},
    {"l": 2, "j": 1.5, "m": [1.5, -1.5], "P": 1},
    {"l": 3, "j": 3.5, "m": [0.5, -0.5], "P": -1},
    {"l": 3, "j": 3.5, "m": [1.5, -1.5], "P": -1},
    {"l": 3, "j": 3.5, "m": [2.5, -2.5], "P": -1},
    {"l": 3, "j": 3.5, "m": [3.5, -3.5], "P": 1},
    {"l": 1, "j": 1.5, "m": [0.5, -0.5], "P": -1},
    {"l": 1, "j": 1.5, "m": [1.5, -1.5], "P": -1},
]

# Assegnamento shell
for i in range(n_rows):
    for j in range(n_cols):
        best_shell = None
        best_score = -1
        for idx, shell in enumerate(shells):
            score = 0
            if shell["l"] == L2[i, j]: score += 1
            if shell["j"] == J2[i, j]: score += 1
            if ms[i, j] in shell["m"]: score += 1
            if shell["P"] == P[i, j]: score += 1
            if score > best_score:
                best_score = score
                best_shell = idx
        assignments[i, j] = best_shell
        scores[i, j] = best_score

print(assignments)
for j in range(n_cols):
    sorted_idxs = np.argsort(assignments[:, j])
    data[:, j] = data[sorted_idxs, j]
# Definizione colori/etichette dinamiche
unique_shells = sorted(set(int(a) for a in np.unique(assignments)))
markers = ['s', '^', 'v', '>', '<', 'p', '*', 'h', 'D', 'o', ',', '.', '1', '2']
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_shells)))

labels = [f"$l={shells[idx]['l']},\ j={shells[idx]['j']}^{{{'+' if shells[idx]['P'] > 0 else '-'}}}$"
          for idx in unique_shells]

# Plotting
x = np.linspace(-0.5, 0.5, n_cols)
fig, ax = plt.subplots(figsize=(6, 8))

for shell_num, color, marker, label in zip(unique_shells, colors, markers, labels):
    mask = (assignments == shell_num)
    for i in range(n_rows):
        if np.any(mask[i, :]):
            ax.plot(x, data[i, :], marker=marker, linestyle='-', color=color, label=label)
            break  # Plotta solo una riga per etichetta legenda

# Etichette e griglia
ax.set_xlabel("$\\beta_2$")
ax.set_ylabel("SPSE of the neutron [MeV]")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc='upper right', fontsize='x-small')
plt.tight_layout()
plt.show()
