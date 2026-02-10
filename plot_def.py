import numpy as np
import matplotlib.pyplot as plt

def roundHI(x):
    """Arrotonda al più vicino mezzo intero."""
    return np.round(x * 2) / 2

# --- Caricamento Dati (invariato) ---
# Assicurati che i percorsi ai file siano corretti
data = np.genfromtxt('output/def_energies.csv')
ms = roundHI(np.genfromtxt('output/m.csv'))
L2 = np.round((np.genfromtxt('output/L2.csv')))
J2 = roundHI((np.genfromtxt('output/J2.csv')))
P = np.round(np.genfromtxt('output/P.csv'))

data = data[:20, :]
ms = ms[:20, :]
L2 = L2[:20, :]
J2 = J2[:20, :]
P = P[:20, :]

n_rows, n_cols = data.shape
assignments = np.full((n_rows, n_cols), -1, dtype=int) # Usiamo -1 per non assegnato
scores = np.zeros((n_rows, n_cols))

# --- Definizione delle Shell (invariato) ---
# Nota: alcune shell sono duplicate. Questo potrebbe essere intenzionale o meno.
# Il codice funzionerà comunque, ma potrebbe plottare linee separate per definizioni identiche.
shells = [
    {"l": 0, "j": 0.5, "m": [0.5, -0.5], "P": 1},   # idx 0
    {"l": 1, "j": 1.5, "m": [1.5, -1.5], "P": -1},  # idx 1
    {"l": 1, "j": 1.5, "m": [0.5, -0.5], "P": -1},  # idx 2
    {"l": 1, "j": 0.5, "m": [0.5, -0.5], "P": -1},  # idx 3
    {"l": 2, "j": 2.5, "m": [0.5, -0.5], "P": 1},   # idx 4
    {"l": 2, "j": 2.5, "m": [1.5, -1.5], "P": 1},   # idx 5
    {"l": 2, "j": 2.5, "m": [2.5, -2.5], "P": 1},   # idx 6
    {"l": 0, "j": 0.5, "m": [0.5, -0.5], "P": 1},   # idx 7 (duplicato di 0)
    {"l": 2, "j": 1.5, "m": [0.5, -0.5], "P": 1},   # idx 6
#    {"l": 2, "j": 1.5, "m": [1.5, -1.5], "P": 1},   # idx 9
    #{"l": 3, "j": 3.5, "m": [0.5, -0.5], "P": -1},  # idx 10
#    {"l": 3, "j": 3.5, "m": [1.5, -1.5], "P": -1},  # idx 11
#    {"l": 3, "j": 3.5, "m": [2.5, -2.5], "P": -1},  # idx 12
#    {"l": 3, "j": 3.5, "m": [3.5, -3.5], "P": 1},   # idx 13
#    {"l": 1, "j": 1.5, "m": [0.5, -0.5], "P": -1},  # idx 14 (duplicato di 2)
#    {"l": 1, "j": 1.5, "m": [1.5, -1.5], "P": -1},  # idx 15 (duplicato di 1)
]

assigned_shells = np.full((len(shells), n_cols), 0, dtype=int) # Inizializzazione
# --- Assegnamento Shell (invariato) ---
for i in range(n_rows):
    for j in range(n_cols):
        best_shell_idx = -1
        best_score = -1
        
        # Stato calcolato corrente
        current_state = {'l': L2[i, j], 'j': J2[i, j], 'm': ms[i, j], 'P': P[i, j]}

        for idx, shell_def in enumerate(shells):
            if assigned_shells[idx, j] >= 2: continue # Non si può assegnare due volte lo stesso stato
            score = 0
            if shell_def["l"] == current_state['l']: score += 1
            if shell_def["j"] == current_state['j']: score += 2 # Diamo più peso a J
            if current_state['m'] in shell_def["m"]: score += 2
            if shell_def["P"] == current_state['P']: score += 1
            
            if score > best_score:
                best_score = score
                if best_shell_idx >= 0:
                    assigned_shells[best_shell_idx, j] -= 1
                best_shell_idx = idx
                assigned_shells[idx, j] += 1
        
        assignments[i, j] = best_shell_idx
        scores[i, j] = best_score

print(assignments)
# =============================================================================
# SEZIONE DI ORDINAMENTO E PLOTTING CORRETTA
# =============================================================================

# NON riordinare la matrice 'data'. L'ordine originale è corretto.
# La riga 'i' non corrisponde alla stessa shell per ogni deformazione.
# Il nostro scopo è USARE la matrice 'assignments' per tracciare ogni shell.

# 1. Preparazione per il plot
x_deformations = np.linspace(-0.5, 0.5, n_cols)
unique_assigned_shells = (np.unique(assignments))

# Rimuoviamo il valore -1 se presente (stati non assegnati)
#if -1 in unique_assigned_shells:
#    unique_assigned_shells.delete(-1)

# Creiamo un dizionario per le etichette per evitare duplicati in legenda
labels_dict = {}
for shell_idx in unique_assigned_shells:
    s = shells[shell_idx]
    # L'etichetta è basata sulle proprietà della shell, non sull'indice
    label_key = (s['l'], s['j'], s['P']) 
    parity_str = '+' if s['P'] > 0 else '-'
    labels_dict[label_key] = f"${s['l']}_{{{s['j']}}}^{parity_str}, m_j = \pm {abs(s['m'][0])}$"


# 2. Creazione del grafico
fig, ax = plt.subplots(figsize=(6, 8))

# Usiamo un ciclo di colori per distinguere le linee
colors = plt.cm.tab20(np.linspace(0, 1.0, len(unique_assigned_shells)))
unique_assigned_shells = unique_assigned_shells[1:8]


# 3. Ciclo di plotting corretto
for i, shell_idx in enumerate(unique_assigned_shells):
    
    # Per questa shell, troviamo la sua energia a ogni step di deformazione
    shell_energies = np.full(n_cols, np.nan) # Inizializza con NaN

    for j in range(n_cols):
        # Trova la riga (l'indice dell'energia) che corrisponde alla nostra shell a questa deformazione
        matching_rows = np.where(assignments[:, j] == shell_idx)[0]
        
        if len(matching_rows) > 0:
            # Se troviamo una corrispondenza, prendiamo la prima (idealmente ce n'è solo una)
            row_index = matching_rows[0]
            shell_energies[j] = data[row_index, j]
            # Se un'altra shell è stata mappata qui, per evitare di riusare lo stesso punto,
            # lo marchiamo come "usato" per questa deformazione j
            #assignments[row_index, j] = -2 

    # Recupera la definizione della shell per l'etichetta
    s = shells[shell_idx]
    label_key = (s['l'], s['j'], s['P'])
    current_label = labels_dict.get(label_key)

    # Plotta la traiettoria di questa shell
    # L'uso di nan permette di avere linee spezzate se uno stato non è stato trovato
    ax.plot(x_deformations, shell_energies, marker='o', markersize=3, linestyle='-', color=colors[i], label=current_label)
    #ax.annotate(current_label, (x_deformations[-1] - 0.2, shell_energies[-1] - 1.1), fontsize=8, color = colors[i])
    
    # Rimuoviamo l'etichetta dal dizionario per non ripeterla nella legenda
    #if current_label is not None:
    #    labels_dict[label_key] = None


# 4. Finalizzazione del grafico
ax.set_xlabel("$\\beta_2$")
ax.set_ylabel("Neutron SPSE [MeV]")
ax.set_title("Neutron SPSE as a function of quadrupole deformation")
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
#ax.set_ylim(np.nanmin(data)-1, np.nanmax(data)+1) # Imposta i limiti in modo intelligente
ax.legend(loc="best", bbox_to_anchor=(1, 0.5), fontsize='small', title="$l_j^P$ $m_j$")
#plt.tight_layout(rect=[0, 0, 0.9, 1]) # Aggiusta il layout per fare spazio alla legenda
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
