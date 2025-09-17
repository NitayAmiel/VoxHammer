import os
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['ATTN_BACKEND'] = 'sdpa'       # Use 'sdpa' to simplify attention capture (xformers supported too)
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import torch
import numpy as np
import trellis.attn_globals as attn_globals
import matplotlib.pyplot as plt
from trellis.modules.sparse import SparseSubdivide, SparseTensor
from trellis.models.sparse_structure_vae import UpsampleBlock3d
from trellis.models.structured_latent_vae.decoder_mesh import SparseSubdivideBlock3d
import torch.nn as nn
from visualizecoords import render_voxel_video_from_L4, render_voxel_set_images

def cat_attn_to_coords(attn, coords):
    idx = coords[:, 1:4].long().to(attn.device)          # shape: (L, 3)
    if torch.any((idx < 0) | (idx >= 64)):
        raise ValueError("Some indices in attn[:,1:4] are out of [0, 63].")
    vals = attn[idx[:, 0], idx[:, 1], idx[:, 2]]    # shape: (L,)
    return torch.cat([coords, vals.unsqueeze(1)], dim=1)

def process_attn(attn_tensor, prompt_attn, target_word, upsample_mode):
    relevant_cols = find_token_columns_for_prompt(pipeline, prompt_attn, target_word)
    head_vecs = attn_tensor[0]  # [16,4096,77]
    valid_heads = ~torch.isnan(head_vecs).any(dim=(1,2))  # [16]
    num_valid = valid_heads.sum().item()
    print(f"Valid heads: {num_valid} out of {len(valid_heads)} (valid indices: {valid_heads.nonzero().squeeze().tolist()})")
    
    if not valid_heads.any():
        print("No valid heads found!")
        return
        
    # Average over valid heads
    valid_attn = head_vecs[valid_heads]  # [valid_heads,4096,77]
    avg_attn = valid_attn.mean(dim=0)  # [4096,77]
    print(f"Valid attention shape: {valid_attn.shape}")
    print(f"Averaged attention shape: {avg_attn.shape}")
    print(f"Attention value range: min={avg_attn.min().item():.6f}, max={avg_attn.max().item():.6f}")
    
    # Process specified columns
    if relevant_cols is None:
        relevant_cols = range(avg_attn.shape[1])
    
    # Create upsampling blocks using UpsampleBlock3d
    upsample = nn.ModuleList([
        UpsampleBlock3d(
            in_channels=1,  # Single channel for attention scores
            out_channels=1,
            mode=upsample_mode  # Use learnable convolution-based upsampling
        ),
        UpsampleBlock3d(
            in_channels=1,
            out_channels=1,
            mode=upsample_mode  # Use learnable convolution-based upsampling
        )
    ]).cuda()
    # pdb.set_trace()
    result = np.zeros((len(relevant_cols), 64, 64, 64), dtype=np.float32)
    for i, col_idx in enumerate(relevant_cols):
        print(f"Processing column {col_idx}")
        # Get column and reshape to 16^3
        col_data = avg_attn[:, col_idx]  # [4096]
        vol_16 = col_data.reshape(16, 16, 16)  # [16,16,16]
        # pdb.set_trace()
        # Convert to dense tensor format for UpsampleBlock3d
        # Shape: [B, C, H, W, D] = [1, 1, 16, 16, 16]
        dense_vol = vol_16.unsqueeze(0).unsqueeze(0).cuda()  # [1, 1, 16, 16, 16]
        
        # Upsample to 64^3 using UpsampleBlock3d
        with torch.no_grad():
            upsampled_vol = dense_vol
            for block in upsample:
                upsampled_vol = block(upsampled_vol)  # Each block doubles the spatial dimensions
                
        # Extract the final volume and convert to numpy
        # After 2 upsampling blocks: 16 -> 32 -> 64
        vol_64 = upsampled_vol.squeeze(0).squeeze(0).cpu().numpy()
        V = vol_64.astype(np.float32).copy()
        vmin, vmax = float(np.nanmin(V)), float(np.nanmax(V))
        print(f"After nan check: vmin={vmin:.6f}, vmax={vmax:.6f}")
        
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            print("Warning: Invalid value range, setting volume to zero")
            V[:] = 0
        else:
            V = (V - vmin) / (vmax - vmin + 1e-8)
            print(f"Normalized value range: [{V.min():.6f}, {V.max():.6f}]")
        result[i] = V
    
    return result
  # [64, 64, 64]

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-large")
pipeline.cuda()
# Run the pipeline
def find_token_columns_for_prompt(pipeline, prompt: str, target_word: str, max_len: int = 77,
                                case_insensitive: bool = True, whole_word: bool = True):
    """Find token columns in attention that correspond to a specific word in the prompt.
    
    Args:
        pipeline: TrellisTextTo3DPipeline instance
        prompt: The full text prompt
        target_word: The specific word to find in the prompt
        max_len: Maximum sequence length (default: 77 for CLIP)
        case_insensitive: Whether to ignore case when matching
        whole_word: Whether to match whole words only
        
    Returns:
        List of column indices corresponding to the target word's tokens
    """
    tokenizer = pipeline.text_cond_model['tokenizer']
    
    # Get tokenizer output with offset mapping - wrap prompt in list to match pipeline's processing
    enc = tokenizer(
        [prompt],  # Wrap in list to match pipeline's processing
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    token_ids = enc.input_ids[0].tolist()
    offsets = enc.offset_mapping[0].tolist()
    
    # Debug: Print tokenization details
    print("\nTokenization Debug:")
    print(f"Prompt: '{prompt}'")
    print(f"Target word: '{target_word}'")
    print("Token mapping:")
    for i, (tid, (start, end)) in enumerate(zip(token_ids, offsets)):
        if end > start:  # Only show non-empty tokens
            token = tokenizer.decode([tid])
            text_span = prompt[start:end] if start < len(prompt) else "<pad>"
            print(f"  Column {i:2d}: Token {tid:5d} = '{token}' (text: '{text_span}', span: {start}:{end})")
    
    # Get special token IDs
    special_ids = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "cls_token_id", "sep_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_ids.add(int(token_id))
    
    # Get valid column indices (exclude special tokens and empty offsets)
    valid_cols = [
        i for i, (tid, (start, end)) in enumerate(zip(token_ids, offsets))
        if (tid not in special_ids) and (end > start)
    ]
    
    # Find target word spans in prompt
    import re
    flags = re.I if case_insensitive else 0
    pattern = rf"\b{re.escape(target_word)}\b" if whole_word else re.escape(target_word)
    spans = [(m.start(), m.end()) for m in re.finditer(pattern, prompt, flags)]
    
    # If no matches with word boundaries, try without
    if not spans and whole_word:
        spans = [(m.start(), m.end()) for m in re.finditer(re.escape(target_word), prompt, flags)]
    
    # Find token columns that overlap with target word spans
    matched_cols = []
    for col_idx in valid_cols:
        start, end = offsets[col_idx]
        for word_start, word_end in spans:
            if not (end <= word_start or start >= word_end):  # Check overlap
                matched_cols.append(col_idx)
                break
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(matched_cols))

prompt_generation = "a table with a lamp and a book."
prompt_attn = prompt_generation
attn_word = "book"
THRESHOLD = 0.8
upsample_mode = "nearest"
outputs = pipeline.generate_and_save_attention(
    prompt_generation=prompt_generation,
    prompt_attn=prompt_attn,
    percentage_of_layers_to_store=1,
)

attn = outputs['attn']
output = outputs['output']
coords = outputs['coords']
# render the original structure
video = render_utils.render_video(output['gaussian'][0])['color']
imageio.mimsave("tmp/attn_gs_org.mp4", video, fps=30)
video = render_utils.render_video(output['mesh'][0])['normal']
imageio.mimsave("tmp/attn_mesh_org.mp4", video, fps=30)

# upsample and gather the relevant columns
attn_processed = process_attn(attn, prompt_attn, attn_word, upsample_mode)
assert len(attn_processed.shape) == 4
for i in range(attn_processed.shape[0]):
    # take tjhe relevant attention values, new coords now is [L,5] columns: [batch, x, y, z, attn]
    output_i, new_coords = pipeline.show_attn(attn_processed[i], threshold=THRESHOLD, original_coords=coords, add_strength=True)
    # output_path = render_voxel_video_from_L4(new_coords, out_path=f"tmp/attn_voxels_{attn_word}_{i}.mp4", use_gpu=True)
    # print(f" saved voxel video to {output_path}")
    render_voxel_set_images(is_gradually=True, voxels_L4=cat_attn_to_coords(torch.from_numpy(attn_processed[i]), coords.cpu())[:, 1:5],out_dir=f"tmp/attn_voxels_{attn_word}_{i}")
    print(f"saved voxel set images to tmp/attn_voxels_{attn_word}_{i}")
    video = render_utils.render_video(output_i['gaussian'][0])['color']
    imageio.mimsave(f"tmp/attn_gs_{attn_word}_{i}.mp4", video, fps=30)
    video = render_utils.render_video(output_i['mesh'][0])['normal']
    imageio.mimsave(f"tmp/attn_mesh_{attn_word}_{i}.mp4", video, fps=30)

