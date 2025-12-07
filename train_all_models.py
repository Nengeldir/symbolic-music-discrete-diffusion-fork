import subprocess
import sys

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
steps_per_eval = "100"
steps_per_sample = "100"
steps_per_checkpoint = "500"

def train_model(model_name):
    print(f"==============================================")
    print(f"Starting training for model: {model_name}")
    print(f"==============================================")
    
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
