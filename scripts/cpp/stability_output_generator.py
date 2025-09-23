import json
import os
import copy
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

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
            json.dump(config_copy, out_f, indent=4, cls=NpEncoder)

        print(f"Creato: {output_path}")


if __name__ == "__main__":
    inputs = np.arange(7, 13, 1)
    print(inputs)
    output_folder = "input/stability"
    title_path = ["outputName"]
    for i in inputs:
        base_config = "input    /generator.json"
        generate_configs(
            base_config,
            output_folder,
            param_path=["box", "size"],
            values=[i],
            title_path=title_path,
            title_prefix=("")
        )

        param_path = ["box", "n"]


        step_sizes = np.arange(0.3, 0.8, 0.1)

        values = i / step_sizes
        values = 2*np.floor(values)
        values = values.astype(np.int32)
        # Genera i file di configurazione
        for j in values:
            generate_configs(
                f"input/stability/_{int(i)}.json",
                output_folder,
                param_path=param_path,
                values=[j],
                title_path=title_path,
                title_prefix=f"run_{int(i)}"
            )

        os.remove(f"input/stability/_{int(i)}.json")
