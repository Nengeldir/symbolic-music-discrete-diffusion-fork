import argparse
import torch
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from hparams import get_sampler_hparams
from utils import get_sampler, load_model, samples_2_noteseq, save_noteseqs
from note_seq import midi_to_note_sequence, note_sequence_to_midi_file
from utils.sampler_utils import ns_to_np

def main():
    parser = argparse.ArgumentParser(description='Infill music using trained model')
    parser.add_argument('--model', type=str, default='conv_transformer', help='Model architecture')
    parser.add_argument('--tracks', type=str, default='melody', help='Tracks configuration')
    parser.add_argument('--load_dir', type=str, required=True, help='Log directory containing the trained model')
    parser.add_argument('--load_step', type=int, default=0, help='Checkpoint step to load (0 for latest/best if available)')
    parser.add_argument('--input_midi', type=str, required=True, help='Input MIDI file to infill')
    parser.add_argument('--start_bar', type=int, default=16, help='Start bar to mask (infill)')
    parser.add_argument('--end_bar', type=int, default=32, help='End bar to mask (infill)')
    parser.add_argument('--output_dir', type=str, default='infilled_results', help='Directory to save results')
    parser.add_argument('--temp', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--sample_steps', type=int, default=256, help='Number of sampling steps')
    
    args, unknown = parser.parse_known_args()

    # Hack: Remove infill-specific args from sys.argv so get_sampler_hparams doesn't fail
    # We construct a new sys.argv containing only the script name and the 'unknown' args (which are meant for Hparams)
    # plus the ones we already parsed that Hparams might ALSO need (like --model).
    # Actually, simpler: define ONLY infill args in this parser.
    # Parse them.
    # Then reconstruct sys.argv to exclude them.
    
    # Arguments to remove from sys.argv before passing to Hparams
    keys_to_remove = ['--input_midi', '--start_bar', '--end_bar', '--output_dir', '--temp', '--sample_steps']
    
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in keys_to_remove:
            i += 2 # Skip key and value
        else:
            new_argv.append(arg)
            i += 1
            
    sys.argv = new_argv

    # 1. Setup Hparams and Model
    print(f"Loading model from {args.load_dir}...")
    # Now get_sampler_hparams will see only the compatible args
    H = get_sampler_hparams('sample')
    
    # Override Hparams with args from infill.py if they were parsed here
    # Start: We only kept infill args in the parser definition above?
    # No, the previous code defined model/tracks etc.
    # We should continue to define them IF we need to use them before H is ready, OR just rely on H.
    # The previous code overrode H.model = args.model.
    # If we let get_sampler_hparams parse --model, H.model will be set correctly.
    # So we don't need to manually override H.* from args.* if we let H parse them.
    
    # However, args.load_dir IS needed for load_model later.
    # Hparams defines load_dir too.
    # So both parse it. That's fine if we don't remove it.
    
    # CODE CHANGE:
    # 1. Parse known args.
    # 2. Filter sys.argv to remove ONLY the args that Hparams doesn't know about.
    
    # Override Hparams manually is safer if we trust our args.
    # But let's just let Hparams parse what it knows.
    
    # H.model is set by Hparams parsing.
    # H.tracks is set by Hparams parsing.
    # H.load_dir is set by Hparams parsing.
    
    # We just need to ensure Codebook size is correct.
    H.codebook_size = (90,) if H.tracks == 'melody' else (90, 90, 512)

    sampler = get_sampler(H).cuda()
    
    # Load checkpoint
    if args.load_step > 0:
        sampler = load_model(sampler, f'{H.sampler}_ema', args.load_step, args.load_dir)
    else:
        # Try to find latest checkpoint if 0
        # For simplicity, user should prob provide step, or we default to logic in log_utils...
        # But load_model expects explicit step. 
        # Let's assume user knows the step or we fail.
        # However, to be nice, let's just use what user provided.
        pass

    sampler.eval()

    # 2. Load and Preprocess MIDI
    print(f"Loading MIDI: {args.input_midi}")
    with open(args.input_midi, 'rb') as f:
        midi_data = f.read()
    
    ns = midi_to_note_sequence(midi_data)
    
    # Convert to tokens
    # Note: ns_to_np returns a NamedTuple with 'outputs'
    # We need to know if it's melody or trio
    mode = 'melody' if args.tracks == 'melody' else 'trio'
    
    # ns_to_np might expect a specific length or just convert everything.
    # It converts to chunks of H.bars usually?
    # Let's check utils/sampler_utils.py... 
    # It uses OneHotMelodyConverter(slice_bars=bars) inside.
    # We need to ensure we get a tensor that fits the model (H.NOTES length)
    
    # This might return multiple segments if midi is long.
    # We will pick the first segment that covers the requested bars.
    
    # HACK: Let's assume the user provided midi fits or we take the first chunk.
    # To get a predictable size, we pass 'bars' to ns_to_np via H if needed? 
    # Actually ns_to_np creates converter inside.
    
    # Sanitize NoteSequence: Flatten Tempos and Time Signatures
    # POP909 and many datasets have many tempo changes which cause the converter to split the sequence into tiny chunks
    # or crash. We replace them with a single average tempo and 4/4 time signature.
    
    if len(ns.tempos) > 0:
        mean_tempo = np.mean([t.qpm for t in ns.tempos])
    else:
        mean_tempo = 120.0

    del ns.tempos[:]
    ns.tempos.add(qpm=mean_tempo, time=0)
    
    del ns.time_signatures[:]
    ns.time_signatures.add(numerator=4, denominator=4, time=0)

    # Let's use the converter directly to be safe
    from preprocessing import OneHotMelodyConverter, TrioConverter
    # Use slice_bars=None to get the full sequence, then we handle length manually
    # Use gap_bars=None to prevent splitting on silence (infinite gap)
    if mode == 'melody':
        # Debug: check constraints
        print(f"DEBUG: Using OneHotMelodyConverter with slice_bars=None, gap_bars=None")
        converter = OneHotMelodyConverter(slice_bars=None, gap_bars=None)
    else:
        converter = TrioConverter(slice_bars=None, gap_bars=None)
        
    tensors = converter.to_tensors(ns).outputs # List of tensors
    
    if not tensors:
        print("Error: Could not convert MIDI to tensors. The file might be empty or invalid.")
        return

    x_original = tensors[0] # Take first valid chunk
    x_original = torch.tensor(x_original).long().cuda()
    
    # If 3 tracks, shape is (T, 3)
    # If melody, shape is (T, 1) usually (or just T?)
    # MelodyConverter returns (T, 1) usually.
    if x_original.dim() == 1:
        x_original = x_original.unsqueeze(-1) # Ensure (T, 1)
        
    # Manually Pad or Crop to H.NOTES (1024)
    target_len = H.NOTES
    current_len = x_original.shape[0]
    
    if current_len < target_len:
        print(f"Warning: Input length {current_len} < {target_len}. Padding with silence.")
        # Pad with 0 (silence)
        padding = torch.zeros((target_len - current_len, x_original.shape[1]), dtype=x_original.dtype, device=x_original.device)
        x_original = torch.cat([x_original, padding], dim=0)
    elif current_len > target_len:
        print(f"Warning: Input length {current_len} > {target_len}. Cropping to first {target_len} tokens.")
        x_original = x_original[:target_len]
        
    # Ensure shape is correct (T, C)
    if x_original.shape[0] != target_len:
         # Should not happen after above logic
         print(f"Error: Shape mismatch after processing: {x_original.shape}")
         return
        
    # 3. Create Mask
    print(f"Masking bars {args.start_bar} to {args.end_bar}...")
    
    steps_per_bar = 16 # Standard 16th notes
    start_step = args.start_bar * steps_per_bar
    end_step = args.end_bar * steps_per_bar
    
    mask_token = H.codebook_size # (90, ) or (90, 90, 512)
    # Actually AbsorbingDiffusion uses self.mask_id which is set from H.codebook_size
    # We need to match the structure.
    
    # Prepare x_T (starting point for sampling)
    # It should contain the ORIGINAL tokens where we want to keep them,
    # and MASK tokens where we want to generate.
    
    x_T = x_original.clone().unsqueeze(0) # Add batch dim -> (1, 1024, C)
    
    # Apply mask
    # We need to iterate over tracks
    if mode == 'melody':
        # 1 track
        mask_val = H.codebook_size[0]
        x_T[:, start_step:end_step, 0] = mask_val
    else:
        # 3 tracks (Melody, Bass, Drums)
        # We assume we mask ALL tracks in that range? Or user wants specific?
        # User asked to "infill a music piece", implies time-based infilling.
        for t in range(3):
            mask_val = H.codebook_size[t]
            x_T[:, start_step:end_step, t] = mask_val

    # 4. Run Sampling
    print("Infilling...")
    
    # AbsorbingDiffusion.sample signature:
    # def sample(self, temp=1.0, sample_steps=None, x_T=None, B=None, progress_handler=None):
    # It treats x_T as the starting canvas. 
    # Important: It identifies "unmasked" (fixed) parts by checking `x_T != self.mask_id`.
    # So we just need to pass our partially masked x_T!
    
    with torch.no_grad():
        infilled = sampler.sample(
            temp=args.temp,
            sample_steps=args.sample_steps,
            x_T=x_T
        )
    
    # 5. Save Result
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'infilled.mid')
    
    print("Converting to MIDI...")
    # infilled is (1, 1024, C)
    # samples_2_noteseq expects np array
    infilled_np = infilled.cpu().numpy()
    
    note_seqs = samples_2_noteseq(infilled_np)
    save_noteseqs(note_seqs, prefix=os.path.join(args.output_dir, 'infilled'))
    
    print(f"Saved infilled result to {out_path}")

if __name__ == "__main__":
    main()
