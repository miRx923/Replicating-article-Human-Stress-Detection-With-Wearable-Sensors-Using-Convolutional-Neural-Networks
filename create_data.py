""" Python script for complete dataset creation. """

# List of scripts to execute
scripts = {
    "Preparing chest script": 'src/creating_data/data_preprocessing/prepare_chest.py',
    "Preparing wrist script": 'src/creating_data/data_preprocessing/prepare_wrist.py',
    "Resampling chest and merging script": 'src/creating_data/data_preprocessing/resample_chest_n_merge.py',
    "Processing signals script": 'src/creating_data/signal_processing/signal_processing.py',
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
        print("Done creating data! Now you can run experiments/run_all_experiments.py script, or wait if you ran main.py.\n")
