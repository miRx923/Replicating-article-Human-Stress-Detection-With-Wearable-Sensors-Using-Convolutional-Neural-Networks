""" Python script for running all experiments from paper. """

# List of scripts to execute
scripts = {
    "S_vs_NS experiments": 'experiments/S_vs_NS.py',
    "B_vs_S_vs_A experiments": 'experiments/B_vs_S_vs_A.py',
    "B_vs_S_vs_A_vs_M experiments": 'experiments/B_vs_S_vs_A_vs_M.py',
}

# Execute each script using exec()
for script_name, script_path in scripts.items():
    print(f"Executing: {script_name}...")

    try:
        with open(script_path, 'r', encoding='utf-8') as file:
            code = file.read()
        exec(code)
        print(f"Finished executing: {script_name}.\n")

    except Exception as e:
        print(f"Error while executing {script_name}: {e}\n")

    finally:
        print("Experiments are done!\n")
