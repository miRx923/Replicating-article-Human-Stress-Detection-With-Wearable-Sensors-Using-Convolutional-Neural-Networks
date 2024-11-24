""" Python script for creating data and running all experiments based on the paper. """

# List of scripts to execute
scripts = {
    "Creating data script": 'create_data.py',
    "Running all experiments script": 'run_all_experiments.py',
}

# Execute each script using exec()
for script_name, script_path in scripts.items():
    print(f"Executing: {script_name}...")

    with open(script_path, 'r', encoding='utf-8') as file:
        code = file.read()
    exec(code)
    print(f"Finished executing: {script_name}.\n")
