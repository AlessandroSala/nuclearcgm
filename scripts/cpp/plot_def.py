import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt('output/def.csv')

def roundHI(x):
    return np.round(x*2)/2

ms = np.genfromtxt('output/m.csv')
ms = roundHI(ms)
print(ms)

data = data[::2]
x = np.linspace(-0.5, 0.5, data.shape[1])
print("Minimum beta: ", x[np.sum(data, axis=0).argmin()])
#np.delete(data, (2), axis=0)
#np.delete(data, (3), axis=0)
print(data.shape[0])
#data = data[np.argsort(data[1:3, 0:5]), 0:5]
data[[1, 2], 6:11] = data[[2, 1], 6:11]
data[[4, 6], 6:11] = data[[6, 4], 6:11]



print(np.sum(data, axis=0))
labels = [r'$0s\ 1/2^+$', r'$0p\ 3/2^-$', r'$1/2_1^-$', r'$0p\ 1/2^-$', r'$0d\ 5/2^+$', r'$1s\ 1/2^+$', r'$0d\ 3/2^+$', r'$3/2_1^+$', r'$1/2_2^-$', r'$1/2_3^+$', r'$3/2_2^+$', r'$1/2_4^+$', r'$5/2_1^+$', r'$3/2_2^+$'] # Aggiungi le tue etichette
labels_corr = [r'$1s\ 1/2^+$', r'$1p\ 3/2^-$', r'$1/2_1^-$', r'$1p\ 1/2^-$',r'$1d\ 5/2^+$', r'$3/2^+$, 'r'$1/2^+$',  r'$3/2_1^+$', r'$1/2_2^-$', r'$1/2_3^+$', r'$3/2_2^+$', r'$1/2_4^+$', r'$5/2_1^+$', r'$3/2_2^+$'] # Aggiungi le tue etichette
# Simboli e colori da utilizzare per ogni traccia (puoi personalizzarli)
markers = ['s', '^', 'v', '>', '<', 'p', '*', 'h', 'D', 'o', ',', '.', '1', '2']
colors = ['black', 'red', 'teal', 'blue', 'purple', 'brown', 'green', 'magenta', 'olive', 'coral', 'cyan', 'darkgoldenrod', 'gray', 'lime']

# Plotta ogni riga dei dati
fig, ax = plt.subplots(figsize=(6, 8))
if data.size > 0:
    for i in range(data.shape[0]):
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