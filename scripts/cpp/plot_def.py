import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt('output/def.csv')

def print_shell(i, j):
    print("Energy: ", data[i, j], " | j: ", J2[i, j], " | m: ", ms[i, j], " | P: ", P[i, j], " | l: ", L2[i, j])

def roundHI(x):
    ceil = np.ceil(x*2)
    floor = np.floor(x*2)
    
    ceil[ceil % 2 == 0] = floor[ceil % 2 == 0]
    return ceil / 2
def momFromSq(x):
    return((0.5*(-1 + np.sqrt(1 + 4*x))))
    

ms = roundHI(np.genfromtxt('output/m.csv'))
L2 = (np.genfromtxt('output/L2.csv'))
J2 = (np.genfromtxt('output/J2.csv'))
P = np.round(np.genfromtxt('output/P.csv'))
#print(J2)
L2 = np.round(momFromSq(L2))
J2 = roundHI(momFromSq(J2))
#exit();
#print(ms)

data = data[::2]
ms = ms[::2]
J2 = J2[::2]
L2 = L2[::2]
P = P[::2]
print(ms)
print(J2)
print(L2)
print(P)
x = np.linspace(-0.5, 0.5, data.shape[1])
print("Minimum beta: ", x[np.sum(data, axis=0).argmin()])
#np.delete(data, (2), axis=0)
#np.delete(data, (3), axis=0)
print(data.shape[0])
#data = data[np.argsort(data[1:3, 0:5]), 0:5]
#data[[1, 2], 6:11] = data[[2, 1], 6:11]
#data[[4, 6], 6:11] = data[[6, 4], 6:11]

shells = [
    {
        "l": 0,
        "j": 0.5,
        "m": [0.5, -0.5],
        "P": 1
    },
    {
        "l": 1,
        "j": 1.5,
        "m": [1.5, -1.5],
        "P": -1,
    },
    {
        "l": 1,
        "j": 1.5,
        "m": [0.5, -0.5],
        "P": -1
    },
    {
        "l": 1,
        "j": 0.5,
        "m": [0.5, -0.5],
        "P": -1
    },
    {
        "l": 2,
        "j": 2.5,
        "m": [0.5, -0.5],
        "P": 1
    },
    {
        "l": 2,
        "j": 2.5,
        "m": [1.5, -1.5],
        "P": 1
    },
    {
        "l": 2,
        "j": 2.5,
        "m": [2.5, -2.5],
        "P": 1
    },
    {
        "l": 0,
        "j": 0.5,
        "m": [0.5, -0.5],
        "P": 1
    },
    {
        "l": 2,
        "j": 1.5,
        "m": [0.5, -0.5],
        "P": 1
    },
    {
        "l": 2,
        "j": 1.5,
        "m": [1.5, -1.5],
        "P": 1
    },
    {
        "l": 3,
        "j": 3.5,
        "m": [0.5, -0.5],
        "P": -1
    },
    {
        "l": 3,
        "j": 3.5,
        "m": [1.5, -1.5],
        "P": -1
    },
    {
        "l": 3,
        "j": 3.5,
        "m": [2.5, -2.5],
        "P": -1
    },
     {
         "l": 3,
         "j": 3.5,
         "m": [3.5, -3.5],
         "P": 1
     },
    {
        "l": 1,
        "j": 1.5,
        "m": [0.5, -0.5],
        "P": -1
    },
    {
        "l": 1,
        "j": 1.5,
        "m": [1.5, -1.5],
        "P": -1
    },

    

]
print("Number of shells: ", len(shells))
print(shells[7])
n_rows, n_cols = data.shape
assignments = np.zeros((n_rows, n_cols))
scores = np.zeros((n_rows, n_cols))

for i in range(n_rows):
    for j in range(n_cols):
        best_shell = None
        best_score = -1
        for idx, shell in enumerate(shells):
            score = 0

            if(shell["l"] == L2[i, j]):
                score += 1
            if(shell["j"] == J2[i, j]):
                score += 1
            if(shell["m"][0] == ms[i, j]):
                score += 1
            if(shell["P"] == P[i, j]):
                score += 1
            if(idx == 3 and i == 3 and j == 1):
                print(score)
                print(best_score)
            if score > best_score:
                best_score = score
                best_shell = idx 

        scores[i, j] = best_score
        assignments[i, j] = best_shell
print(assignments)
print("Scores:")
print(scores)

print_shell(6, 0)
#exit()
#data = data[:7]
#assignments = assignments[:7]

for i in range(n_cols):
    sorted_idxs = np.argsort(assignments[:, i])
    data[:, i] = data[sorted_idxs, i]
#exit()

print(np.sum(data, axis=0))

print(np.argsort(assignments, axis = 0))
data = data[:7]
assignments = assignments[:7]
print(data)
#exit()

labels = [r'$0s\ 1/2^+$', r'$0p\ 3/2^-$', r'$1/2_1^-$', r'$0p\ 1/2^-$', r'$0d\ 5/2^+$', r'$1s\ 1/2^+$', r'$0d\ 3/2^+$', r'$3/2_1^+$', r'$1/2_2^-$', r'$1/2_3^+$', r'$3/2_2^+$', r'$1/2_4^+$', r'$5/2_1^+$', r'$3/2_2^+$'] # Aggiungi le tue etichette
labels_corr = [r'$1s\ 1/2^+$', r'$1p\ 3/2^-$', r'$1/2_1^-$', r'$1p\ 1/2^-$',r'$1d\ 5/2^+$', r'$3/2^+$, 'r'$1/2^+$',  r'$3/2_1^+$', r'$1/2_2^-$', r'$1/2_3^+$', r'$3/2_2^+$', r'$1/2_4^+$', r'$5/2_1^+$', r'$3/2_2^+$'] # Aggiungi le tue etichette
# Simboli e colori da utilizzare per ogni traccia (puoi personalizzarli)
markers = ['s', '^', 'v', '>', '<', 'p', '*', 'h', 'D', 'o', ',', '.', '1', '2']
colors = ['black', 'red', 'teal', 'blue', 'purple', 'brown', 'green', 'magenta', 'olive', 'coral', 'cyan', 'darkgoldenrod', 'gray', 'lime']

# Plotta ogni riga dei dati
fig, ax = plt.subplots(figsize=(6, 8))
if data.size > 0:
    for i in range(7):
        ax.plot(x, data[i, :], marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], label=labels_corr[i])
        plt.text(0.3, data[i, 8] + 0.5, labels_corr[i], color=colors[i % len(colors)], fontsize=10, ha='left')


    # Aggiungi etichette e titolo
    ax.set_xlabel("$\\beta_2$") # Sostituisci con la tua etichetta per l'asse x
    ax.set_ylabel('SPSE of the neutron [MeV]') # Mantieni l'etichetta dell'asse y come nel tuo grafico

    # Aggiungi la legenda
    #ax.legend(loc='upper right', fontsize='small', frameon=False)

    # Imposta i limiti degli assi se necessario
    # ax.set_xlim(-0.6, 0.6)
    # ax.set_ylim(min_y, max_y) # Sostituisci con i tuoi limiti y

    # Aggiungi delle annotazioni come nel tuo grafico (opzionale)
    # ax.text(x_posizione, y_posizione, 'testo', color='colore', fontsize=dimensione)

    # Mostra il grafico
    plt.grid(True, which='major', linestyle='--', alpha=0.6) # Aggiungi una griglia se lo desideri
    plt.tight_layout() # Regola il layout per evitare sovrapposizioni
    plt.show()