import argparse
import numpy as np
import os
from utils import samples_2_noteseq, save_noteseqs

def main():
    parser = argparse.ArgumentParser(description='Convert NPZ samples to MIDI')
    parser.add_argument('--file', type=str, required=True, help='Path to input .npz file')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        return

    print(f"Loading {args.file}...")
    try:
        data = np.load(args.file, allow_pickle=True)
        # np.load might return the array directly or a NpzFile object depending on how it was saved
        # The training script uses np.save, so it should be a direct array if loaded with allow_pickle=True
        # But commonly np.save saves a single array.
        
        # If saved with np.savez, it would be a dict. np.save saves a single .npy inside .npz usually?
        # Actually np.save saves .npy. np.save_z saves .npz. 
        # train.py uses np.save(..., samples, ...) but extension is .npz manually added?
        # log_utils.py: np.save(log_dir + f'/samples_{step}.npz', np_samples, allow_pickle=True)
        # If it's np.save(file.npz, arr), it's actually a .npy format just with wrong extension.
        # So np.load should load it as an array.
        
        if isinstance(data, np.lib.npyio.NpzFile):
            # Just in case it was actually savez
            samples = data['arr_0']
        else:
            samples = data

        print(f"Loaded samples with shape: {samples.shape}")
        
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # Create output directory
    input_dir = os.path.dirname(args.file)
    basename = os.path.basename(args.file).replace('.npz', '')
    output_dir = os.path.join(input_dir, f"{basename}_midi")
    os.makedirs(output_dir, exist_ok=True)

    print("Converting to MIDI...")
    # samples_2_noteseq expects numpy array (B, T, Tracks)
    # It returns a list of NoteSequences
    try:
        note_seqs = samples_2_noteseq(samples)
        
        for i, ns in enumerate(note_seqs):
            out_path = os.path.join(output_dir, f"sample_{i}.mid")
            save_noteseqs([ns], prefix=os.path.join(output_dir, f"sample"))
            # convert save_noteseqs takes logic:
            # def save_noteseqs(ns, prefix='pre_adv'):
            #     for i, n in enumerate(ns):
            #         note_sequence_to_midi_file(n, prefix + f'_{i}.mid')
            # So we should just call it once with the list
            
        save_noteseqs(note_seqs, prefix=os.path.join(output_dir, basename))
        print(f"Saved {len(note_seqs)} MIDI files to {output_dir}")
        
    except Exception as e:
        print(f"Error converting/saving: {e}")

if __name__ == "__main__":
    main()
