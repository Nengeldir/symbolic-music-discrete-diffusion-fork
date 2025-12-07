
import numpy as np
import argparse
import os
import sys
import torch
import random
from note_seq import note_sequence_to_midi_file

# Add current directory to path
sys.path.append(os.getcwd())

from preprocessing.data import OneHotMelodyConverter, TrioConverter

def main():
    parser = argparse.ArgumentParser(description='Inspect prepared training data (NPY to MIDI)')
    parser.add_argument('--file', type=str, required=True, help='Path to input .npy file (e.g. data/POP909_melody.npy)')
    parser.add_argument('--mode', type=str, default='melody', choices=['melody', 'trio'], help='Data mode')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to inspect')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        return

    print(f"Loading {args.file}...")
    try:
        data = np.load(args.file, allow_pickle=True)
        # Data is a list of tensors (or list of list of tensors?)
        # prepare_data.py: result = list(chain(*result)) -> list of tensors (or tuples of tensors if trio?)
        # For melody: list of (T, 1) or similar.
        print(f"Loaded data with {len(data)} items.")
        
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # Helper function to convert tensor back to MIDI
    if args.mode == 'melody':
        # Re-instantiate converter just to use 'from_tensors'
        converter = OneHotMelodyConverter(slice_bars=None, gap_bars=None) 
    else:
        converter = TrioConverter(slice_bars=None, gap_bars=None)
        
    # Select random samples
    if len(data) == 0:
        print("Data is empty.")
        return
        
    sample_indices = random.sample(range(len(data)), min(args.samples, len(data)))
    
    output_dir = "inspected_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Selecting {len(sample_indices)} random samples...")
    
    for idx in sample_indices:
        item = data[idx]
        
        # Prepare for from_tensors
        # from_tensors expects a LIST of samples (batch).
        # item might be a single tensor or tuple.
        # TrioConverter.from_tensors expects sample shape (T, 3) probably.
        # OneHotMelodyConverter.from_tensors expects sample shape (T, 1) or (T,).
        
        # Check shape
        if isinstance(item, np.ndarray):
            print(f"Sample {idx} shape: {item.shape}")
            # If (T, 1) remove dim if needed?
            # from_tensors calls decode_event on "s".
            # For melody converter: s = input_sample.
            
            # Let's wrap in list [item]
            # MelodyConverter requires (T, D) or (T,)? 
            # Looking at code: s = sample. if end_token... decode_event(e)
            # It iterates over s. So s should be 1D array of indices?
            # Or one-hot?
            # Converter output is usually indices if saved as npy?
            # No, converter to_tensors returns one-hot usually if output_dtype=bool?
            # Wait, `OneHotMelodyConverter` outputs one-hot vectors?
            # preprocessing/data.py line 566: inputs=seqs (np_onehot)
            # So the saved data is One-Hot encoded (Bools or Floats).
            # But wait, `npz_to_midi` expected samples.
            
            # If the data is One-Hot, `from_tensors` usually handles it?
            # MelodyConverter.from_tensors (inherited from LegacyEventList):
            # s = sample (arg).
            # if end_token... s = s[:index]
            # for e in s: decode_event(e)
            # decode_event expects an index?
            # "s = sample # np.argmax(sample, axis=-1)" is commented out in line 572 of data.py!
            # So it expects `sample` to already be indices?
            # Or maybe `to_tensors` returned indices?
            
            # Let's check `to_tensors` in `LegacyEventListOneHotConverter` (data.py line 560):
            # seqs.append(np_onehot(...))
            # np_onehot (line 129) returns `np.expand_dims(np.array(indices), 1)`
            # That is NOT one-hot. That is indices with an extra dim!
            # The docstring says "Converts 1D array of indices to a one-hot 2D array" but code does `expand_dims`.
            # Wait, line 131-133 commented out the actual one-hot logic.
            # So it returns (T, 1) array of INDICES.
            
            # So `from_tensors` expects `s` to be (T, 1) or (T,)?
            # Line 579: `for e in s:`
            # If s is (T, 1), e is array([index]).
            # decode_event probably expects int.
            
            # Let's flatten item if it's (T, 1)
            # item_flat = item.flatten()
            
            ns_list = converter.from_tensors([item.flatten()])
            ns = ns_list[0]
            
            out_path = os.path.join(output_dir, f"inspected_{args.mode}_{idx}.mid")
            note_sequence_to_midi_file(ns, out_path)
            print(f"Saved {out_path} (Notes: {len(ns.notes)})")
            
        else:
            print(f"Sample {idx} is not ndarray, it's {type(item)}")

if __name__ == "__main__":
    main()
