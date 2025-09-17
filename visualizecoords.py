import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio
from io import BytesIO
from typing import Optional, Tuple
import os
import math
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def render_voxel_set_images(
    *,
    # ---- Mode selection & data ----
    is_gradually: bool = False,
    voxels_L4: np.ndarray | None = None,  # [L,4] => [x,y,z,S]
    set_xyz: np.ndarray | None = None,    # [L+M,3]
    subset_xyz: np.ndarray | None = None, # [L,3] or None
    # ---- Shared view/render params ----
    grid_size: int = 64,
    num_views: int = 8,
    out_dir: str = "renders",
    elev_deg: float = 22.5,
    azim_start: float = 0.0,
    azim_end: float = 360.0,
    edgecolor=None,           # e.g., "k" for outlines
    show_axes: bool = False,  # hide axes for clean images
    dpi: int = 200,
    bg: str = "white",
    # ---- Gradual-color (S) options ----
    vmin: float | None = None,
    vmax: float | None = None,
    # ---- Set-vs-Subset color options ----
    subset_color=(1.0, 0.2, 0.2, 1.0),   # RGBA (red-ish)
    set_only_color=(0.2, 0.4, 1.0, 0.9), # RGBA (blue-ish)
):
    """
    Renders X (=num_views) images from different azimuth angles.

    Modes:
      1) Gradual color (green->red by S):
         Triggered if is_gradually=True or subset_xyz is None.
         Requires voxels_L4 with shape [L,4] columns [x,y,z,S].

      2) Set vs Subset coloring:
         Triggered if is_gradually=False and subset_xyz is not None.
         Requires set_xyz [L+M,3] and subset_xyz [L,3].

    Outputs:
      Saves PNGs into out_dir named view_XX.png
    """
    os.makedirs(out_dir, exist_ok=True)
    azims = np.linspace(azim_start, azim_end, num=num_views, endpoint=False)

    # ---------------------------
    # Mode 1: Gradual color by S
    # ---------------------------
    if is_gradually or subset_xyz is None:
        if voxels_L4 is None:
            raise ValueError("Gradual mode requires voxels_L4=[L,4] with [x,y,z,S].")

        voxels_L4 = np.asarray(voxels_L4)
        if voxels_L4.ndim != 2 or voxels_L4.shape[1] != 4:
            raise ValueError("voxels_L4 must have shape [L,4] with columns [x,y,z,S].")

        coords = voxels_L4[:, :3].astype(np.int64)
        S = voxels_L4[:, 3].astype(float)

        if np.any(coords < 0) or np.any(coords >= grid_size):
            raise ValueError("Some voxel indices are outside [0, grid_size-1].")

        # Normalize S → [0,1] for color mapping (green→red)
        s_min = np.min(S) if vmin is None else float(vmin)
        s_max = np.max(S) if vmax is None else float(vmax)
        if math.isclose(s_max, s_min):
            s_norm = np.zeros_like(S)
        else:
            s_norm = np.clip((S - s_min) / (s_max - s_min), 0.0, 1.0)

        # Occupancy and facecolors
        occ = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
        facecolors = np.zeros(occ.shape + (4,), dtype=float)
        # Map: low->green (0,1,0), high->red (1,0,0)
        rgba = np.zeros((coords.shape[0], 4), dtype=float)
        rgba[:, 0] = s_norm
        rgba[:, 1] = 1.0 - s_norm
        rgba[:, 2] = 0.0
        rgba[:, 3] = 1.0

        x, y, z = coords.T
        occ[x, y, z] = True
        facecolors[x, y, z] = rgba
        coords = occ

    # --------------------------------
    # Mode 2: Set vs Subset coloring
    # --------------------------------
    else:
        if set_xyz is None or subset_xyz is None:
            raise ValueError("Set-vs-Subset mode requires set_xyz and subset_xyz, each [*,3].")

        subset_xyz = np.asarray(subset_xyz, dtype=np.int64)
        set_xyz = np.asarray(set_xyz, dtype=np.int64)

        if subset_xyz.ndim != 2 or subset_xyz.shape[1] != 3:
            raise ValueError("subset_xyz must have shape [L, 3]")
        if set_xyz.ndim != 2 or set_xyz.shape[1] != 3:
            raise ValueError("set_xyz must have shape [L+M, 3]")

        # Clip to grid bounds
        subset_xyz = np.clip(subset_xyz, 0, grid_size - 1)
        set_xyz = np.clip(set_xyz, 0, grid_size - 1)

        # Build occupancy volumes
        set_vol = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
        subset_vol = np.zeros_like(set_vol, dtype=bool)
        set_vol[set_xyz[:, 0], set_xyz[:, 1], set_xyz[:, 2]] = True
        subset_vol[subset_xyz[:, 0], subset_xyz[:, 1], subset_xyz[:, 2]] = True
        subset_vol &= set_vol  # ensure subset within set
        union = set_vol

        # Colors
        colors = np.zeros(union.shape + (4,), dtype=float)
        colors[subset_vol] = subset_color
        colors[union & ~subset_vol] = set_only_color
        coords = union
        facecolors = colors

    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg)

    ax.set_xlim(0, grid_size); ax.set_ylim(0, grid_size); ax.set_zlim(0, grid_size)
    ax.set_box_aspect([1, 1, 1])

    if not show_axes:
        ax.set_axis_off()

    ax.voxels(coords, facecolors=facecolors, edgecolor=edgecolor)
    for i, az in enumerate(azims):
        ax.view_init(elev=elev_deg, azim=az)

        out_path = os.path.join(out_dir, f"view_{i:02d}_az{int(az)}.png")
        plt.savefig(out_path, bbox_inches="tight", facecolor=bg, pad_inches=0.05)
        print(f"saved {i}/{num_views}")
    plt.close(fig)

    return {"mode": "set_vs_subset", "out_dir": out_dir, "num_images": num_views}

def render_voxel_video_from_L4(
    L4: np.ndarray,
    grid_shape: Optional[Tuple[int, int, int]] = None,
    n_frames: int = 90,
    elev: float = 30.0,
    azim_start: float = 0.0,
    azim_end: float = 360.0,
    figsize: Tuple[int, int] = (6, 6),
    out_path: str = "voxels.mp4",
    dpi: int = 100,
    use_gpu: bool = True,
):
    """
    Render a rotating video from an [L,5] array of active voxel indices with strengths.
    Column 0 is ignored (all zeros). Columns 1..3 are integer x,y,z. Column 4 is strength.

    Parameters
    ----------
    L4 : np.ndarray          shape [L,5] (note: still called L4 for compatibility)
    grid_shape : (nx,ny,nz)  optional, inferred from max coords if None
    n_frames : int           number of frames for a full rotation
    elev : float             elevation angle (deg)
    azim_start : float       starting azimuth (deg)
    azim_end : float         ending azimuth (deg)
    figsize : (w,h)          matplotlib figure size (inches)
    out_path : str           output MP4 path
    dpi : int                figure DPI (resolution)
    use_gpu : bool           if True, try GPU rendering (much faster), else use CPU
    """
    
    # Try GPU rendering first if requested and available
    if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            print("🚀 Using GPU rendering (should be much faster)")
            return _render_voxel_video_gpu(L4, grid_shape, n_frames, out_path)
        except Exception as e:
            print(f"⚠️  GPU rendering failed ({e}), falling back to CPU")
    elif use_gpu and not TORCH_AVAILABLE:
        print("⚠️  PyTorch not available, using CPU rendering")
    elif use_gpu:
        print("⚠️  CUDA not available, using CPU rendering")
    
    print("🔄 Using CPU rendering (matplotlib - will be slower)")
    
    # CPU rendering (original implementation)
    # Handle both numpy arrays and torch tensors
    if TORCH_AVAILABLE and isinstance(L4, torch.Tensor):
        L4 = L4.detach().cpu().numpy()
    L4 = np.asarray(L4)
    assert L4.ndim == 2 and L4.shape[1] == 5, "Input must be shape [L,5]"
    coords = L4[:, 1:4].astype(int)
    strengths = L4[:, 4].astype(float)
    assert np.all(coords >= 0), "Voxel indices must be non-negative integers"
    
    # Normalize strengths to [0, 1]
    if len(strengths) > 0:
        strengths_min = strengths.min()
        strengths_max = strengths.max()
        if strengths_max > strengths_min:
            strengths_normalized = (strengths - strengths_min) / (strengths_max - strengths_min)
        else:
            strengths_normalized = np.ones_like(strengths)  # All same strength
    else:
        strengths_normalized = np.array([])

    # infer grid size if needed - default to 64^3
    if grid_shape is None:
        if coords.size > 0:
            max_xyz = coords.max(axis=0)
            # Use 64^3 as default, but expand if coordinates exceed this
            grid_shape = tuple(max(64, max_xyz[i] + 1) for i in range(3))
        else:
            grid_shape = (64, 64, 64)
    nx, ny, nz = grid_shape

    # build volume with strength-based colors
    vol = np.zeros((nx, ny, nz), dtype=bool)
    vol_colors = np.zeros((nx, ny, nz, 4), dtype=float)  # RGBA colors
    
    inb = (coords[:,0] < nx) & (coords[:,1] < ny) & (coords[:,2] < nz)
    coords_valid = coords[inb]
    strengths_valid = strengths_normalized[inb]
    
    vol[coords_valid[:,0], coords_valid[:,1], coords_valid[:,2]] = True
    
    # Create colors based on strength: fixed blue color with varying transparency
    for i, (coord, strength) in enumerate(zip(coords_valid, strengths_valid)):
        x, y, z = coord
        # Fixed blue color with transparency based on strength
        red = 0.0
        green = 0.3
        blue = 1.0
        alpha = 0.2 + 0.8 * strength  # More opacity for stronger voxels (0.2 to 1.0)
        vol_colors[x, y, z] = [red, green, blue, alpha]

    xs, ys, zs = np.where(vol)
    if xs.size == 0:
        raise ValueError("No active voxels found in the provided array.")

    # bounds for equal aspect (so it doesn’t look stretched)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    z_min, z_max = zs.min(), zs.max()
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) + 1

    def set_equal_aspect(ax):
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        z_mid = 0.5 * (z_min + z_max)
        half = max_range / 2.0
        ax.set_xlim(x_mid - half, x_mid + half)
        ax.set_ylim(y_mid - half, y_mid + half)
        ax.set_zlim(z_mid - half, z_mid + half)

    # write video with optimizations and good frame rate
    writer = imageio.get_writer(out_path, fps=30, codec="libx264", quality=8, macro_block_size=1)
    
    # Copy EXACT camera movement from render_utils.py
    # This is exactly how render_utils.py does it:
    yaws = np.linspace(0, 2 * np.pi, n_frames)  # azimuth angles
    pitchs = 0.25 + 0.5 * np.sin(np.linspace(0, 2 * np.pi, n_frames))  # elevation in radians
    # Convert pitch from radians to degrees for matplotlib
    elevation_angles = np.degrees(pitchs)  # This gives the perfect viewing angles
    azimuth_angles = np.degrees(yaws)      # Convert to degrees
    
    try:
        for i, (az, el) in enumerate(zip(azimuth_angles, elevation_angles)):
            # Create figure with optimized settings
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=el, azim=np.degrees(az))  # Convert radians to degrees
            ax.set_axis_off()

            # render voxels with strength-based colors
            ax.voxels(vol, facecolors=vol_colors, edgecolor=None)

            set_equal_aspect(ax)

            # Optimize image saving
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, 
                       facecolor='black', edgecolor='none')
            plt.close(fig)  # Important: close figure to free memory
            
            buf.seek(0)
            frame = imageio.v3.imread(buf.getvalue())
            writer.append_data(frame)
            
            # Print progress for long renders
            if i % max(1, n_frames // 10) == 0:
                print(f"Rendering frame {i+1}/{n_frames}")
    finally:
        writer.close()

    return out_path


def _render_voxel_video_gpu(L4, grid_shape: Optional[Tuple[int, int, int]], n_frames: int, out_path: str) -> str:
    """Simple GPU-accelerated voxel rendering using PyTorch."""
    
    device = torch.device('cuda')
    print(f"🚀 GPU rendering {L4.shape[0]} voxels with {n_frames} frames")
    
    # Process coordinates - keep on GPU if already there
    if isinstance(L4, torch.Tensor):
        if L4.device != device:
            L4 = L4.to(device)
        coords = L4[:, 1:4].int()
        strengths = L4[:, 4].float()
    else:
        L4 = torch.from_numpy(np.asarray(L4)).to(device)
        coords = L4[:, 1:4].int()
        strengths = L4[:, 4].float()
    
    # Normalize strengths to [0, 1]
    if len(strengths) > 0:
        strengths_min = strengths.min()
        strengths_max = strengths.max()
        if strengths_max > strengths_min:
            strengths_normalized = (strengths - strengths_min) / (strengths_max - strengths_min)
        else:
            strengths_normalized = torch.ones_like(strengths)  # All same strength
    else:
        strengths_normalized = torch.tensor([], device=device)
    
    # Grid shape handling
    if grid_shape is None:
        if coords.numel() > 0:
            max_xyz = coords.max(dim=0)[0]  # torch.max returns (values, indices)
            grid_shape = tuple(max(64, int(max_xyz[i]) + 1) for i in range(3))
        else:
            grid_shape = (64, 64, 64)
    
    # Normalize coordinates (already on GPU)
    max_dim = max(grid_shape)
    points = (coords.float() / max_dim) * 2.0 - 1.0  # Normalize to [-1, 1]
    
    # Copy EXACT camera movement from render_utils.py
    yaws = torch.linspace(0, 2 * torch.pi, n_frames, device=device)
    pitchs = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * torch.pi, n_frames, device=device))
    # These are already in radians, perfect for our rotation matrices
    azimuth_angles = yaws
    elevation_angles = pitchs
    
    frames = []
    resolution = 512  # Fixed resolution for GPU version
    
    for i, (azimuth, elevation) in enumerate(zip(azimuth_angles, elevation_angles)):
        # Rotation around Z axis (azimuth) with varying elevation
        cos_a, sin_a = torch.cos(azimuth), torch.sin(azimuth)
        cos_e, sin_e = torch.cos(elevation), torch.sin(elevation)
        
        # Combined rotation: first around Z (azimuth), then elevation
        rotation_z = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], device=device, dtype=torch.float32)
        
        rotation_x = torch.tensor([
            [1, 0, 0],
            [0, cos_e, -sin_e],
            [0, sin_e, cos_e]
        ], device=device, dtype=torch.float32)
        
        # Combined rotation matrix
        rotation_matrix = torch.matmul(rotation_x, rotation_z)
        
        # Rotate points
        rotated_points = torch.matmul(points, rotation_matrix.T)
        
        # Simple orthographic projection - keep it simple like matplotlib
        # Just project X and Y coordinates, no complex perspective
        screen_coords_x = ((rotated_points[:, 0] + 1) * (resolution / 2)).long()
        screen_coords_y = ((rotated_points[:, 1] + 1) * (resolution / 2)).long()
        screen_coords = torch.stack([screen_coords_x, screen_coords_y], dim=1)
        screen_coords = torch.clamp(screen_coords, 5, resolution - 6)  # Leave room for cube drawing
        
        # Use Z for depth sorting only
        z_coords = rotated_points[:, 2]
        
        # Sort by depth (furthest first for proper occlusion)
        depth_order = torch.argsort(z_coords, descending=True)
        screen_coords_sorted = screen_coords[depth_order]
        strengths_sorted = strengths_normalized[depth_order]
        
        # Create frame with WHITE background (like matplotlib version)
        frame = torch.ones((resolution, resolution, 3), device=device)  # White background
        
        # Draw voxels as 3D-looking cubes
        if len(screen_coords_sorted) > 0:
            x_coords = screen_coords_sorted[:, 0]
            y_coords = screen_coords_sorted[:, 1]
            
            # Fixed blue color for all voxels (transparency will show strength)
            red_component = torch.zeros_like(strengths_sorted)      # No red
            green_component = torch.full_like(strengths_sorted, 0.3) # Small green 
            blue_component = torch.ones_like(strengths_sorted)       # Full blue
            
            # Create color tensor [N, 3] - all voxels same blue color
            voxel_colors = torch.stack([red_component, green_component, blue_component], dim=1)
            
            # Calculate alpha (transparency) based on strength
            alpha_values = 0.2 + 0.8 * strengths_sorted  # 0.2 (weak) to 1.0 (strong)
            
            # Draw voxels as 3x3 cubes (fast but visible)
            if len(x_coords) > 0:
                # Apply alpha blending vectorized
                white_bg = torch.tensor([1.0, 1.0, 1.0], device=device)
                final_colors = alpha_values.unsqueeze(1) * voxel_colors + (1 - alpha_values.unsqueeze(1)) * white_bg
                
                # Draw 3x3 cubes for each voxel
                cube_size = 3
                for i in range(len(x_coords)):
                    x_center = x_coords[i]
                    y_center = y_coords[i]
                    color = final_colors[i]
                    
                    # Draw 3x3 cube
                    x_start = max(0, x_center - cube_size//2)
                    x_end = min(resolution, x_center + cube_size//2 + 1)
                    y_start = max(0, y_center - cube_size//2)
                    y_end = min(resolution, y_center + cube_size//2 + 1)
                    
                    frame[y_start:y_end, x_start:x_end] = color
        
        # Convert to numpy and add to frames
        frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame_np)
        
        if i % max(1, n_frames // 10) == 0:
            print(f"  GPU frame {i+1}/{n_frames}")
    
    # Save video with good frame rate
    import os
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    imageio.mimsave(out_path, frames, fps=30)  # Good frame rate for smooth viewing
    
    print(f"✅ GPU video saved to: {out_path}")
    return out_path
