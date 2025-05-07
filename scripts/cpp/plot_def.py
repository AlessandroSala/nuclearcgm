import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt('output/def.csv')
data = data[::2]
np.delete(data, (2), axis=0)
np.delete(data, (3), axis=0)

data[data[:, 5].argsort()]

print(data)
x = np.linspace(-0.5, 0.5, data.shape[1])
labels = [r'$0s\ 1/2^+$', r'$0p\ 3/2^-$', r'$1/2_1^-$', r'$0p\ 1/2^-$', r'$0d\ 5/2^+$', r'$1s\ 1/2^+$', r'$0d\ 3/2^+$', r'$3/2_1^+$', r'$1/2_2^-$', r'$1/2_3^+$', r'$3/2_2^+$', r'$1/2_4^+$', r'$5/2_1^+$', r'$3/2_2^+$'] # Aggiungi le tue etichette
# Simboli e colori da utilizzare per ogni traccia (puoi personalizzarli)
markers = ['s', '^', 'v', '>', '<', 'p', '*', 'h', 'D', 'o', ',', '.', '1', '2']
colors = ['black', 'red', 'teal', 'blue', 'purple', 'brown', 'green', 'magenta', 'olive', 'coral', 'cyan', 'darkgoldenrod', 'gray', 'lime']

# Plotta ogni riga dei dati
fig, ax = plt.subplots(figsize=(6, 8))
if data.size > 0:
    for i in range(data.shape[0]):
        ax.plot(x, data[i, :], marker=markers[i % len(markers)], linestyle='-', color=colors[i % len(colors)], label=labels[i])

    # Aggiungi etichette e titolo
    ax.set_xlabel('Asse X') # Sostituisci con la tua etichetta per l'asse x
    ax.set_ylabel('SPSE of the proton [MeV]') # Mantieni l'etichetta dell'asse y come nel tuo grafico
    ax.set_title('Titolo del Grafico') # Aggiungi il tuo titolo

    # Aggiungi la legenda
    ax.legend(loc='upper right', fontsize='small', frameon=False)

    # Imposta i limiti degli assi se necessario
    # ax.set_xlim(-0.6, 0.6)
    # ax.set_ylim(min_y, max_y) # Sostituisci con i tuoi limiti y

    # Aggiungi delle annotazioni come nel tuo grafico (opzionale)
    # ax.text(x_posizione, y_posizione, 'testo', color='colore', fontsize=dimensione)

    # Mostra il grafico
    plt.grid(True, which='major', linestyle='--', alpha=0.6) # Aggiungi una griglia se lo desideri
    plt.tight_layout() # Regola il layout per evitare sovrapposizioni
    plt.show()