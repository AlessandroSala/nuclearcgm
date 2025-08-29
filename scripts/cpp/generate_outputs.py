import json
import os
import copy

def set_nested_value(d, keys, value):
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def generate_configs(base_config_path, output_dir, param_path, values, title_path=None, title_prefix="Simulation"):
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for val in values:
        config_copy = copy.deepcopy(base_config)

        set_nested_value(config_copy, param_path, val)

        if title_path is not None:
            set_nested_value(config_copy, title_path, f"{title_prefix}_{val}")

        param_str = "_".join(param_path)
        output_filename = f"{title_prefix}_{val}.json"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w') as out_f:
            json.dump(config_copy, out_f, indent=4)

        print(f"Creato: {output_path}")


if __name__ == "__main__":
    base_config = "input/exec/input.json"
    output_folder = "input/steps"

    param_path = ["box", "n"]

    title_path = ["outputName"]

    # Lista di valori per steps
    steps_values = [100, 200, 500, 1000]

    # Genera i file di configurazione
    generate_configs(
        base_config,
        output_folder,
        param_path=param_path,
        values=steps_values,
        title_path=title_path,
        title_prefix="run"
    )
