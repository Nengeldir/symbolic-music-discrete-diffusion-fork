import subprocess
import sys

import os
import re

# Models to train
models = ["conv_transformer", "transformer", "hierarch_transformer", "U_transformer"]

# Common arguments
dataset = "data/POP909_melody.npy"
bars = "64"
batch_size = "4"
tracks = "melody"
# Note: If epochs is set (!= None), it overrides train_steps
epochs = "100"
train_steps = "100000"
steps_per_log = "10"
steps_per_eval = "1000"
steps_per_sample = "1000"
steps_per_checkpoint = "500"

def get_latest_checkpoint(model_name):
    # Calculate NOTES as per default_hparams.py logic
    # bars is a string in the global vars, convert to int
    n_bars = int(bars)
    notes = n_bars * 16
    
    # Construct log directory name
    log_dir_name = f"log_{model_name}_{tracks}_{notes}"
    log_dir_path = os.path.join("logs", log_dir_name, "saved_models")
    
    if not os.path.exists(log_dir_path):
        return 0, log_dir_name
        
    max_step = 0
    # Pattern to match checkpoint files, e.g., absorbing_500.th
    # Note: The model_save_name in save_model (log_utils.py) seems to depend on hparams.sampler
    # In default_hparams.py, sampler is "absorbing" for all these models.
    # So files will look like absorbing_{step}.th
    pattern = re.compile(r"absorbing_(\d+)\.th")
    
    for filename in os.listdir(log_dir_path):
        match = pattern.match(filename)
        if match:
            step = int(match.group(1))
            if step > max_step:
                max_step = step
                
    return max_step, log_dir_name

def train_model(model_name):
    print(f"==============================================")
    print(f"Starting training for model: {model_name}")
    print(f"==============================================")
    
    load_step, load_dir = get_latest_checkpoint(model_name)
    
    cmd = [
        sys.executable, "train.py",
        "--dataset", dataset,
        "--bars", bars,
        "--batch_size", batch_size,
        "--tracks", tracks,
        "--model", model_name,
        "--epochs", epochs,
        "--train_steps", train_steps,
        "--steps_per_log", steps_per_log,
        "--steps_per_eval", steps_per_eval,
        "--steps_per_sample", steps_per_sample,
        "--steps_per_checkpoint", steps_per_checkpoint,
        "--amp",
    ]
    
    if load_step > 0:
        print(f"Resuming from checkpoint: step {load_step} in {load_dir}")
        cmd.extend(["--load_step", str(load_step)])
        cmd.extend(["--load_dir", load_dir])
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Finished training for model: {model_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to train model: {model_name}")
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    print("----------------------------------------------")

if __name__ == "__main__":
    for model in models:
        train_model(model)

    print("All training tasks completed.")
