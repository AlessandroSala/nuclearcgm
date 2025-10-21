import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_json_data(file_path):

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            data = data["data"]
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON.")
        return

    # Extract Beta, Eint, EpairN, EpairP
    betas = [item['betaReal'] for item in data if 'betaReal' in item]
    eints = [item['Eint'] for item in data if 'Eint' in item]
    econst = [item['constraints_energy'] for item in data if 'constraints_energy' in item]
    epairNs = [item['EpairN'] for item in data if 'EpairN' in item]
    epairPs = [item['EpairP'] for item in data if 'EpairP' in item]

    try:
        with open('output/mg_final_curve_no_j/mg_curve_no_j.json', 'r') as f:
            datanoJ = json.load(f)
            datanoJ = datanoJ["data"]
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON.")
        return

    betasnoJ = [item['betaReal'] for item in datanoJ if 'betaReal' in item]
    eintsnoJ = [item['Eint'] for item in datanoJ if 'Eint' in item]


    energies_hfbtho = np.genfromtxt('plots/deformation/energies_hfbtho.csv')
    betas_hfbtho = np.genfromtxt('plots/deformation/betas_hfbtho.csv')

    min_beta_arg = find_nearest(betas_hfbtho, min(betas))
    max_beta_arg = find_nearest(betas_hfbtho, max(betas))+1
    betas_hfbtho = betas_hfbtho[min_beta_arg:max_beta_arg]
    energies_hfbtho = energies_hfbtho[min_beta_arg:max_beta_arg]

    if not betas or not eints:
        print("Error: The JSON file does not contain 'Eint' or 'beta' data.")
        return
    if not epairNs or not epairPs:
        print("Warning: The JSON file does not contain both 'EpairN' and 'EpairP' data. Only plotting Eint.")
        epair_sum = None
    else:
        epair_sum = - (np.array(epairNs) + np.array(epairPs))  # negative sum

    # Sort data by Beta
    if epair_sum is not None:
        sorted_data = sorted(zip(betas, eints, epair_sum))
        sorted_betas = np.array([b for b, _, _ in sorted_data])
        sorted_eints = np.array([e for _, e, _ in sorted_data])
        sorted_epair_sum = np.array([p for _, _, p in sorted_data])
    else:
        sorted_data = sorted(zip(betas, eints))
        sorted_betas = np.array([b for b, _ in sorted_data])
        sorted_eints = np.array([e for _, e in sorted_data])

    # Create interpolation for Eint
    x_new = np.linspace(sorted_betas.min(), sorted_betas.max(), 500)
    #eint_spline = make_interp_spline(sorted_betas, sorted_eints, k=3)
    #y_eint_smooth = eint_spline(x_new)

    # Create stacked plots
    plt.style.use('seaborn-v0_8-ticks')
    show_pairing = False

    if show_pairing:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8),
                                       gridspec_kw={'height_ratios': [1, 2]},
                                       sharex=True)
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(6, 6), sharex = True)

    fontsize = 12
    # Top plot: -(EpairN + EpairP)
    if epair_sum is not None and show_pairing:
        ax1.plot(sorted_betas, sorted_epair_sum, color='teal', label='-Epair', alpha=0.7, linestyle='-')
        #ax1.plot(x_new, y_epair_smooth, '-', color='teal', linewidth=2, label='-Epair (spline)')
        ax1.set_ylabel(r'$- E_\text{pair} [MeV]$', fontsize=13)
        #ax1.legend(fontsize=fontsize, frameon=True, fancybox=True, shadow=True)
        ax1.set_ylim(np.min(sorted_epair_sum) -0.5, max(sorted_epair_sum)-10 )
        ax1.grid(True, linestyle='--', linewidth=0.5)
        ax1.set_title(r"$^{20}$Ne", fontsize=18, fontweight='bold', pad=20)
        ax1.tick_params(axis = 'both', which = 'both', labelsize = fontsize)

    # Bottom plot: Eint
    ax2.plot(sorted_betas, sorted_eints, color='crimson', label='Eint data', alpha=0.7, linestyle='-')
    ax2.plot(betas_hfbtho, energies_hfbtho, color='teal', label='E HFTHO', alpha=0.7, linestyle='-')
    ax2.plot(betasnoJ, eintsnoJ, color='green', label='E no J', alpha=0.7, linestyle='-')
    #ax2.plot(x_new, y_eint_smooth, '-', color='crimson', linewidth=2, label='Eint (spline)')
    ax2.set_xlabel(r"$\beta_2$", fontsize=14, labelpad=15)
    ax2.set_ylabel('E (MeV)', fontsize=13)
    #ax2.legend(fontsize=fontsize, frameon=True, fancybox=True, shadow=True)
    ax2.set_ylim(min(eintsnoJ)-0.5, max(sorted_eints) + 1)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.tick_params(axis = 'both', which = 'both', labelsize = fontsize)
    emin_idx = np.argmin(sorted_eints)
    emin = sorted_eints[emin_idx]
    bmin = sorted_betas[emin_idx]

    emin_idx_hfbtho = np.argmin(energies_hfbtho)
    emin_hfbtho = energies_hfbtho[emin_idx_hfbtho]
    bmin_hfbtho = betas_hfbtho[emin_idx_hfbtho]
    emin_idx_noJ = np.argmin(eintsnoJ)
    emin_noJ = eintsnoJ[emin_idx_noJ]
    bmin_noJ = betasnoJ[emin_idx_noJ]

    marker_size = 4
    marker_style = 'D'
    plt.plot(bmin_hfbtho, emin_hfbtho,marker_style, color='teal', markersize=marker_size)
    plt.plot(bmin, emin, marker_style, color='crimson', markersize=marker_size)
    plt.plot(bmin_noJ, emin_noJ, marker_style, color='green', markersize=marker_size)
    print("beta min GCG no j: ", bmin_noJ)

    ax2.text(0.7, 0.95,
             "$\\boldsymbol{HFBTHO}$" "\n" fr"$E_{{min}} = {emin_hfbtho:.2f}$ MeV" "\n" fr"$\beta_2 = {bmin_hfbtho:.3f}$",
             transform=ax2.transAxes,
             fontsize=11,
             verticalalignment='top',
             horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='teal', alpha=0.8))
# Add small legend box
    ax2.text(0.3, 0.95,
             "$\\boldsymbol{GCG}$" "\n" fr"$E_{{min}} = {emin:.2f}$ MeV" "\n" fr"$\beta_2 = {bmin:.3f}$",
             transform=ax2.transAxes,
             fontsize=11,
             verticalalignment='top',
             horizontalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='crimson', alpha=0.8))
    # Clean layout
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == '__main__':
    #plot_json_data('output/def_pairing_save/mg.json')
    plot_json_data('output/mg_deformation_curve_final/test.json')
    
