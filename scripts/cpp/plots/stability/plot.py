import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === PARAMETRI ===
folder = "./data"   # Cartella dove sono i JSON
E_ref = -128.400
n_steps = 5                    # Numero di step desiderati

# === RACCOLTA DEI DATI ===
data = []

# Scansiona tutti i file .json nella cartella
for filename in os.listdir(folder):
    if filename.endswith(".json"):
        filepath = os.path.join(folder, filename)
        with open(filepath, "r") as f:
            try:
                content = json.load(f)
                eint = content.get("Eint", None)
                a = content.get("a", None)
                step = content.get("step", None)
                
                if eint is not None and a is not None and step is not None:
                    diff = abs((eint / E_ref)*100-100)
                    data.append([a, step, diff])
            except json.JSONDecodeError:
                print(f"Errore nel file: {filename}")

# Converte in DataFrame Pandas
df = pd.DataFrame(data, columns=["a", "step", "diff"])

# === CREAZIONE DI 3 LIVELLI DI STEP ===
# Divide i valori di step in 3 bin e prende la media di ogni bin come valore rappresentativo
df["step_bin"] = pd.qcut(df["step"], q=n_steps, duplicates="drop")
df["step_center"] = df["step_bin"].apply(lambda x: x.mid)

# === CREAZIONE DELLA MATRICE PER LA HEATMAP ===
pivot_table = df.pivot_table(
    index="step_center", columns="a", values="diff", aggfunc=np.mean
)
pivot_table.index = pivot_table.index.round(2)

# === PLOT ===
plt.figure(figsize=(6, 6))
hm = sns.heatmap(
    pivot_table,
    cmap="RdYlGn_r",
    cbar_kws={'label': r"Relative error [-]"},
    annot_kws={"color": "#222222"},
    fmt=".6f",
    annot = True
)
hm.set_yticklabels([f"{y:.2f}" for y in pivot_table.index])
plt.xlabel("Box size [fm]")
plt.ylabel("Step size [fm]")
plt.title("Stability map $^{16}$O")
plt.tight_layout()
plt.show()

