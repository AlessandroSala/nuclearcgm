import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_json_data(file_path):
    """
    Reads a JSON file, extracts 'Eint', 'EpairN', 'EpairP', and 'beta' data,
    and plots two stacked graphs:
      - Top: -(EpairN + EpairP) vs beta
      - Bottom: Eint vs beta
    """

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON.")
        return

    # Extract Beta, Eint, EpairN, EpairP
    betas = [item['beta'] for item in data if 'beta' in item]
    eints = [item['Eint'] for item in data if 'Eint' in item]
    epairNs = [item['EpairN'] for item in data if 'EpairN' in item]
    epairPs = [item['EpairP'] for item in data if 'EpairP' in item]

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
    eint_spline = make_interp_spline(sorted_betas, sorted_eints, k=3)
    y_eint_smooth = eint_spline(x_new)

    if epair_sum is not None:
        epair_spline = make_interp_spline(sorted_betas, sorted_epair_sum, k=3)
        y_epair_smooth = epair_spline(x_new)

    # Create stacked plots
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8),
                                   gridspec_kw={'height_ratios': [1, 2]},
                                   sharex=True)

    fontsize = 12
    # Top plot: -(EpairN + EpairP)
    if epair_sum is not None:
        ax1.plot(sorted_betas, sorted_epair_sum, 'x', color='teal', label='-Epair', alpha=0.7, linestyle='--')
        #ax1.plot(x_new, y_epair_smooth, '-', color='teal', linewidth=2, label='-Epair (spline)')
        ax1.set_ylabel(r'$- E_\text{pair} (MeV)$', fontsize=13)
        #ax1.legend(fontsize=fontsize, frameon=True, fancybox=True, shadow=True)
        ax1.set_ylim(min(sorted_epair_sum) -0.5, max(sorted_epair_sum) + 1)
        ax1.grid(True, linestyle='--', linewidth=0.5)
        ax1.set_title(r"$^{20}$Ne", fontsize=18, fontweight='bold', pad=20)
        ax1.tick_params(axis = 'both', which = 'both', labelsize = fontsize)

    # Bottom plot: Eint
    ax2.plot(sorted_betas, sorted_eints, 'x', color='crimson', label='Eint data', alpha=0.7, linestyle='--')
    #ax2.plot(x_new, y_eint_smooth, '-', color='crimson', linewidth=2, label='Eint (spline)')
    ax2.set_xlabel(r"$\beta_2$", fontsize=14, labelpad=15)
    ax2.set_ylabel('E (MeV)', fontsize=13)
    #ax2.legend(fontsize=fontsize, frameon=True, fancybox=True, shadow=True)
    ax2.set_ylim(min(sorted_eints)-0.5, max(sorted_eints) + 5)
    ax2.grid(True, linestyle='--', linewidth=0.5)
    ax2.tick_params(axis = 'both', which = 'both', labelsize = fontsize)

    # Clean layout
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == '__main__':
    #plot_json_data('output/def_pairing_save/mg.json')
    plot_json_data('output/def_pairing_si_coul/si.json')
