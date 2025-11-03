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
    betas = [item['beta'] for item in data if 'beta' in item]
    eints = [item['Eint'] for item in data if 'Eint' in item]
    econst = [item['constraints_energy'] for item in data if 'constraints_energy' in item]
    epairNs = [item['EpairN'] for item in data if 'EpairN' in item]
    epairPs = [item['EpairP'] for item in data if 'EpairP' in item]

#   try:
#       with open('output/mg_final_curve_no_j/mg_curve_no_j.json', 'r') as f:
#           datanoJ = json.load(f)
#           datanoJ = datanoJ["data"]
#   except FileNotFoundError:
#       print(f"Error: The file at {file_path} was not found.")
#       return
#   except json.JSONDecodeError:
#       print(f"Error: The file at {file_path} is not a valid JSON.")
#       return

#   betasnoJ = [item['betaReal'] for item in datanoJ if 'betaReal' in item]
#   eintsnoJ = [item['Eint'] for item in datanoJ if 'Eint' in item]


    energies_hfbtho = np.genfromtxt('plots/deformation/ev8_en_nopair.csv')
    betas_hfbtho = np.genfromtxt('plots/deformation/ev8_betas_nopair.csv')
    epairNev8 = np.genfromtxt('plots/deformation/epairN_ev8.csv')
    epairPev8 = np.genfromtxt('plots/deformation/epairP_ev8.csv')

    min_beta_arg = find_nearest(betas_hfbtho, min(betas))
    max_beta_arg = find_nearest(betas_hfbtho, max(betas))+1
    betas_hfbtho = betas_hfbtho[min_beta_arg:max_beta_arg]
    energies_hfbtho = energies_hfbtho[min_beta_arg:max_beta_arg]
    epairNev8 = epairNev8[min_beta_arg:max_beta_arg]
    epairPev8 = epairPev8[min_beta_arg:max_beta_arg]

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

    fontsize = 12
    # Top plot: -(EpairN + EpairP)
    if epair_sum is not None and show_pairing:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8),
                                       gridspec_kw={'height_ratios': [1, 2]},
                                       sharex=True)
        ax1.plot(sorted_betas, sorted_epair_sum, color='crimson', label='-Epair', alpha=0.7, linestyle='-')
        ax1.plot(betas_hfbtho, -(epairNev8+epairPev8), color='teal', label='-Epair', alpha=0.7, linestyle='-')
        #ax1.plot(x_new, y_epair_smooth, '-', color='teal', linewidth=2, label='-Epair (spline)')
        ax1.set_ylabel(r'$- E_\text{pair} [MeV]$', fontsize=13)
        #ax1.legend(fontsize=fontsize, frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, linestyle='--', linewidth=0.5)
        ax1.set_title(r"$^{24}$Mg", fontsize=18, fontweight='bold', pad=20)
        ax1.tick_params(axis = 'both', which = 'both', labelsize = fontsize)
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(6, 6), sharex = True)

    beta_smooth = np.linspace(sorted_betas.min(), sorted_betas.max(), 200)
    spline = make_interp_spline(sorted_betas, sorted_eints, k=3)
    Eint_smooth = spline(beta_smooth)

    # Bottom plot: Eint
    #ax2.plot(sorted_betas, sorted_eints, color='crimson', label='Eint data', alpha=0.7, linestyle='-')
    ax2.plot(beta_smooth, Eint_smooth, color='crimson', label='Eint data', alpha=0.7, linestyle='-')
    ax2.plot(betas_hfbtho, energies_hfbtho, color='teal', label='E HFTHO', alpha=0.7, linestyle='-')
    #ax2.plot(betasnoJ, eintsnoJ, color='green', label='E no J', alpha=0.7, linestyle='-')
    #ax2.plot(x_new, y_eint_smooth, '-', color='crimson', linewidth=2, label='Eint (spline)')
    ax2.set_xlabel(r"$\beta_2$", fontsize=14, labelpad=15)
    ax2.set_ylabel('E (MeV)', fontsize=13)
    #ax2.legend(fontsize=fontsize, frameon=True, fancybox=True, shadow=True)
    ax2.set_ylim(min(eints)-0.5, max(energies_hfbtho) + 3)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.tick_params(axis = 'both', which = 'both', labelsize = fontsize)
    emin_idx = np.argmin(sorted_eints)
    emin = sorted_eints[emin_idx]
    bmin = sorted_betas[emin_idx]

    emin_idx_hfbtho = np.argmin(energies_hfbtho)
    emin_hfbtho = energies_hfbtho[emin_idx_hfbtho]
    bmin_hfbtho = betas_hfbtho[emin_idx_hfbtho]
    #emin_idx_noJ = np.argmin(eintsnoJ)
    #emin_noJ = eintsnoJ[emin_idx_noJ]
    #bmin_noJ = betasnoJ[emin_idx_noJ]

    marker_size = 4
    marker_style = 'D'
    plt.plot(bmin_hfbtho, emin_hfbtho,marker_style, color='teal', markersize=marker_size )
    plt.plot(bmin, emin, marker_style, color='crimson', markersize=marker_size)
    #plt.plot(bmin_noJ, emin_noJ, marker_style, color='green', markersize=marker_size)

    ax2.text(0.7, 0.95,
             "$\\boldsymbol{EV8}$" "\n" fr"$E_{{min}} = {emin_hfbtho:.2f}$ MeV" "\n" fr"$\beta_2 = {bmin_hfbtho:.3f}$",
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
    print(sorted_betas)
    # Clean layout
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == '__main__':
    #plot_json_data('output/def_pairing_save/mg.json')
    plot_json_data('output/mg_compare_ev8_nopair/run.json')
    
