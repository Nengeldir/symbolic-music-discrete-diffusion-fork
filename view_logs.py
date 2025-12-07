import argparse
import torch
import os
import visdom
import re
import numpy as np
import sys

# Add current directory to path so we can import utils
sys.path.append(os.getcwd())
from utils.log_utils import vis_samples, log

def get_latest_file(base_dir, subdir, prefix):
    target_dir = os.path.join(base_dir, subdir)
    if not os.path.exists(target_dir):
        return None, -1
    
    files = os.listdir(target_dir)
    # Filter files that match prefix and end with extension (optional, but good for safety)
    target_files = [f for f in files if f.startswith(prefix)]
    
    if not target_files:
        return None, -1
        
    def extract_step(filename):
        # Extract the last number found in the filename
        match = re.search(r'_(\d+)\.', filename)
        return int(match.group(1)) if match else -1
        
    target_files.sort(key=extract_step)
    latest_file = os.path.join(target_dir, target_files[-1])
    return latest_file, extract_step(target_files[-1])

def main():
    parser = argparse.ArgumentParser(description='View training logs in Visdom')
    parser.add_argument('--log_dir', type=str, required=True, help='Path to log directory (e.g., logs/log_conv_transformer_melody_1024)')
    parser.add_argument('--port', type=int, default=8097, help='Visdom port')
    args = parser.parse_args()

    print(f"Connecting to Visdom on port {args.port}...")
    try:
        vis = visdom.Visdom(port=args.port, use_incoming_socket=False, raise_exceptions=True)
        if not vis.check_connection():
             raise Exception("Not connected")
    except Exception as e:
        print(f"Error connecting to Visdom: {e}")
        print("Make sure the server is running: python -m visdom.server")
        return

    # Load Stats
    latest_stats, _ = get_latest_file(args.log_dir, 'saved_stats', 'stats_')
    if latest_stats:
        print(f"Loading stats from {latest_stats}...")
        stats = torch.load(latest_stats, map_location=torch.device('cpu'))

        mean_losses = stats.get('mean_losses')
        val_losses = stats.get('val_losses')
        elbo = stats.get('elbo')
        val_elbos = stats.get('val_elbos')
        steps_per_log = stats.get('steps_per_log', 10)
        steps_per_eval = stats.get('steps_per_eval', 100)

        if mean_losses is not None and len(mean_losses) > 0:
            print(f"Plotting Loss ({len(mean_losses)} points)...")
            x_axis = np.arange(0, len(mean_losses)) * steps_per_log
            vis.line(mean_losses, x_axis, win='loss', opts=dict(title='Loss'))

        if elbo is not None and len(elbo) > 0:
            print(f"Plotting ELBO ({len(elbo)} points)...")
            x_axis = np.arange(0, len(elbo)) * steps_per_log
            vis.line(elbo, x_axis, win='ELBO', opts=dict(title='ELBO'))

        if val_losses is not None and len(val_losses) > 0:
            print(f"Plotting Validation Loss ({len(val_losses)} points)...")
            x_axis = np.arange(1, len(val_losses) + 1) * steps_per_eval 
            vis.line(val_losses, x_axis, win='Val_loss', opts=dict(title='Validation Loss'))

        if val_elbos is not None and len(val_elbos) > 0:
            print(f"Plotting Validation ELBO ({len(val_elbos)} points)...")
            x_axis = np.arange(1, len(val_elbos) + 1) * steps_per_eval
            vis.line(val_elbos, x_axis, win='Val_elbo', opts=dict(title='Validation ELBO'))
    else:
        print(f"No stats files found in {args.log_dir}/saved_stats")

    # Load Samples
    # Note: log_utils.py saves samples in "samples" dir with name "samples_{step}.npz"
    latest_samples_file, step = get_latest_file(args.log_dir, 'samples', 'samples_')
    if latest_samples_file:
        print(f"Loading samples from {latest_samples_file} (Step {step})...")
        try:
            # np.load might return NpzFile or array depending on save method
            data = np.load(latest_samples_file, allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile):
                samples = data['arr_0']
            else:
                samples = data
            
            print(f"Visualizing samples (Shape: {samples.shape})...")
            # vis_samples handles audio generation and sending to visdom
            vis_samples(vis, samples, step)
            print("Samples sent to Visdom.")
        except Exception as e:
            print(f"Failed to visualize samples: {e}")
    else:
        print(f"No sample files found in {args.log_dir}/samples")

    print("Done! Check your Visdom dashboard (usually http://localhost:8097).")

if __name__ == "__main__":
    main()
