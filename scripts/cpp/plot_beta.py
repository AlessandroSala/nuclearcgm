import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_json_data(file_path):
    """
    Reads a JSON file, extracts 'Eint' and 'Beta' data, and plots them.
    The function also fits a smooth curve to the data points.

    Args:
        file_path (str): The path to the JSON file.
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

    # Extract Eint and Beta values from the list of dictionaries
    betas = [item['beta'] for item in data]
    eints = [item['Eint'] for item in data]

    # Check if we have data to plot
    if not betas or not eints:
        print("Error: The JSON file does not contain 'Eint' or 'Beta' data.")
        return

    # Sort data based on Beta for correct interpolation
    sorted_data = sorted(zip(betas, eints))
    sorted_betas = np.array([item[0] for item in sorted_data])
    sorted_eints = np.array([item[1] for item in sorted_data])

    # Create a smooth interpolation curve
    x_new = np.linspace(sorted_betas.min(), sorted_betas.max(), 500)
    spline = make_interp_spline(sorted_betas, sorted_eints, k=3)  # k=3 for a cubic spline
    y_smooth = spline(x_new)

    # Plotting the data
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    # Plot data points
    plt.plot(sorted_betas, sorted_eints, 'o', label='Data', markersize=8, color='crimson', alpha=0.7)

    # Adding titles and labels
    plt.title(r"$^{24}$Mg total energy as a function of $\beta_2$", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel(r"$\beta_2$", fontsize=14, labelpad=15)
    plt.ylabel('E', fontsize=14, labelpad=15)

    # Add a legend and grid
    plt.legend(fontsize=12, frameon=True, shadow=True, fancybox=True)

    # Customize the ticks
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Add annotations for a cleaner look
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Make the plot layout tight
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage:
# Assuming your JSON file is named 'data.json' and is in the same directory.
# Example content of 'data.json':
# [
#     {"Eint": 10.5, "Beta": 1.1},
#     {"Eint": 12.3, "Beta": 1.5},
#     {"Eint": 15.0, "Beta": 2.0},
#     {"Eint": 18.2, "Beta": 2.5},
#     {"Eint": 22.0, "Beta": 3.0}
# ]
if __name__ == '__main__':
    plot_json_data('output/def/surf_beta_mg.json')
