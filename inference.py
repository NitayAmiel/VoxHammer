import argparse
import os
from typing import Optional, Tuple
import torch

from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from voxhammer.bpy_render import render_3d_model
from voxhammer.delete_region_voxel import process_delete_ply
from voxhammer.edit_pipeline import run_edit, run_edit_text
from voxhammer.extract_feature import extract_features

PROMPT_3D_EDIT = "a dog in a yellow raincoat with boots"
PROMPT_SRC = "a dull in a yellow raincoat with boots"
def run_3d_rendering(input_model_path: str, render_dir: str, **render_kwargs) -> dict:
    """
    Step 1: Render 3D model to generate multi-view images

    Args:
        input_model_path: Path to input 3D model file
        render_dir: Directory to save rendered images
        **render_kwargs: Additional rendering parameters

    Returns:
        Dictionary containing rendering results
    """
    print("=" * 50)
    print("STEP 1: 3D Model Rendering")
    print("=" * 50)

    if os.path.exists(os.path.join(render_dir, "transforms.json")) and os.path.exists(
        os.path.join(render_dir, "mesh.ply")
    ):
        print(f"Render directory {render_dir} already exists")
        return {
            "rendered": True,
            "num_views": 150,
            "output_dir": render_dir,
            "transforms_file": os.path.join(render_dir, "transforms.json"),
            "mesh_file": os.path.join(render_dir, "mesh.ply"),
        }

    # Default rendering parameters
    default_params = {
        "num_views": 150,
        "scale": 1.0,
        "offset": None,
        "resolution": 512,
        "engine": "CYCLES",
        "geo_mode": False,
        "split_normal": False,
        "save_mesh": True,
    }

    # Update with provided parameters
    default_params.update(render_kwargs)

    print(f"Input model: {input_model_path}")
    print(f"Output directory: {render_dir}")
    print(f"Rendering parameters: {default_params}")

    result = render_3d_model(
        file_path=input_model_path, output_dir=render_dir, **default_params
    )
    print(f"Rendering completed successfully!")
    print(f"Generated {result['num_views']} views")
    print(f"Transforms file: {result['transforms_file']}")
    if result["mesh_file"]:
        print(f"Mesh file: {result['mesh_file']}")
    return result


def run_feature_extraction(render_dir: str, **feature_kwargs) -> dict:
    """
    Step 2: Extract features from rendered images

    Args:
        render_dir: Directory containing rendered images
        **feature_kwargs: Additional feature extraction parameters

    Returns:
        Dictionary containing feature extraction results
    """
    print("=" * 50)
    print("STEP 2: Feature Extraction")
    print("=" * 50)

    # Default feature extraction parameters
    default_params = {"model": "dinov2_vitl14_reg", "batch_size": 10}

    # Update with provided parameters
    default_params.update(feature_kwargs)

    print(f"Render directory: {render_dir}")
    print(f"Feature extraction parameters: {default_params}")

    extract_features(render_dir, **default_params)
    features_path = os.path.join(render_dir, "features.npz")
    print(f"Feature extraction completed successfully!")
    print(f"Features saved to: {features_path}")
    return {"features_path": features_path}


def run_voxel_masking(mask_glb_path: str, render_dir: str, **mask_kwargs) -> dict:
    """
    Step 3: Generate voxel mask for editing

    Args:
        mask_glb_path: Path to mask GLB file (defines the region to be edited)
        render_dir: Directory containing render outputs
        **mask_kwargs: Additional masking parameters

    Returns:
        Dictionary containing masking results
    """
    print("=" * 50)
    print("STEP 3: Voxel Masking")
    print("=" * 50)

    # Default masking parameters
    default_params = {"filter_method": "volume", "voxel_size": 1 / 64}

    # Update with provided parameters
    default_params.update(mask_kwargs)

    print(f"Mask GLB file: {mask_glb_path}")
    print(f"Render directory: {render_dir}")
    print(f"Masking parameters: {default_params}")

    process_delete_ply(mask_glb_path, render_dir, **default_params)
    voxels_delete_path = os.path.join(render_dir, "voxels_delete.ply")
    print(f"Voxel masking completed successfully!")
    print(f"Mask file: {voxels_delete_path}")
    return {"mask_path": voxels_delete_path}


def run_single_ss_step(
    pipeline,
    noise_input: torch.Tensor,
    tgt_prompt: str,
    t_step: int,
    order: Optional[int] = None,
    pos: Optional[int] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Run a single step of the Sparse Structure (SS) sampling process and save attention maps.

    Args:
        pipeline: The TRELLIS pipeline instance
        noise_input: Input noise tensor for sampling (should match model's expected shape)
        tgt_prompt: Target prompt for conditioning
        t_step: Current timestep for sampling
        order: Optional order parameter for attention saving
        pos: Optional position parameter for attention saving

    Returns:
        Tuple containing:
        - Output tensor from the sampling step
        - Dictionary of saved attention maps
    """
    from voxhammer.edit_pipeline import run_once_and_save_attn
    
    run_feature_extraction(render_dir)
    # Run one step and collect outputs
    output = run_once_and_save_attn(
        pipeline=pipeline,
        noise_input=noise_input,
        tgt_prompt=tgt_prompt,
        t_step=t_step,
        order=order,
        pos=pos
    )
    
    return output

def run_3d_editing(
    pipeline, render_dir: str, image_dir: str, output_path: str, use_text: bool = False, src_prompt: str = PROMPT_SRC, tgt_prompt: str = PROMPT_3D_EDIT, **edit_kwargs
) -> dict:
    """
    Step 4: Perform 3D editing using TRELLIS pipeline

    Args:
        render_dir: Directory containing render outputs and features
        image_dir: Directory containing source, target, and mask images
        output_path: Path for final output GLB file
        use_text: Whether to use text for editing
        src_prompt: Source prompt for text editing
        tgt_prompt: Target prompt for text editing
        **edit_kwargs: Additional editing parameters

    Returns:
        Dictionary containing editing results
    """
    print("=" * 50)
    print("STEP 4: 3D Editing")
    print("=" * 50)

    # Default editing parameters
    default_params = {"skip_step": 0, "re_init": False, "cfg": [5.0, 6.0, 0.0, 0.0]}

    # Update with provided parameters
    default_params.update(edit_kwargs)

    print(f"Render directory: {render_dir}")
    print(f"Image directory: {image_dir}")
    print(f"Output path: {output_path}")
    print(f"Editing parameters: {default_params}")

    # Check required files exist
    required_files = [
        os.path.join(render_dir, "voxels.ply"),
        os.path.join(render_dir, "features.npz"),
        os.path.join(render_dir, "voxels_delete.ply")
    ]
    if not use_text:
        required_files.append(os.path.join(image_dir, "2d_render.png"))
        required_files.append(os.path.join(image_dir, "2d_edit.png"))
        required_files.append(os.path.join(image_dir, "2d_mask.png"))

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    if use_text and not src_prompt or not tgt_prompt:
        raise ValueError("Source and target prompts are required for text editing")

    try:
        if use_text:
            results = run_edit_text(pipeline, render_dir, src_prompt, tgt_prompt, output_path, **default_params)
        else:
            results = run_edit(pipeline, render_dir, image_dir, output_path, **default_params)        
    except Exception as e:
        print(f"3D editing failed: {e}")
        raise


def run_complete_pipeline(
    pipeline,
    input_model_path: str,
    mask_glb_path: str,
    render_dir: str,
    image_dir: str,
    output_path: str,
    use_text: bool = False,
    src_prompt: str = PROMPT_SRC,
    tgt_prompt: str = PROMPT_3D_EDIT,
    render_params: Optional[dict] = None,
    feature_params: Optional[dict] = None,
    mask_params: Optional[dict] = None,
    edit_params: Optional[dict] = None,
) -> dict:
    """
    Run the complete 3D editing pipeline

    Args:
        input_model_path: Path to input 3D model file
        mask_glb_path: Path to mask GLB file (defines the region to be edited)
        render_dir: Directory for render outputs
        image_dir: Directory containing source, target, and mask images
        output_path: Path for final output GLB file (must end with .glb)
        render_params: Parameters for 3D rendering step
        feature_params: Parameters for feature extraction step
        mask_params: Parameters for voxel masking step
        edit_params: Parameters for 3D editing step

    Returns:
        Dictionary containing results from all steps
    """
    print("=" * 60)
    print("STARTING COMPLETE 3D EDITING PIPELINE")
    print("=" * 60)

    # Validate output_path extension
    if not output_path.lower().endswith(".glb"):
        raise ValueError("output_path must end with .glb extension")

    # Initialize results dictionary
    results = {
        "input_model": input_model_path,
        "mask_glb": mask_glb_path,
        "render_dir": render_dir,
        "image_dir": image_dir,
        "final_output": output_path,
    }

    # Step 1: 3D Rendering
    render_results = run_3d_rendering(
        input_model_path, render_dir, **(render_params or {})
    )
    results["rendering"] = render_results

    # Step 2: Feature Extraction
    feature_results = run_feature_extraction(render_dir, **(feature_params or {}))
    results["features"] = feature_results

    # Step 3: Voxel Masking
    mask_results = run_voxel_masking(mask_glb_path, render_dir, **(mask_params or {}))
    results["masking"] = mask_results

    # Step 4: 3D Editing
    edit_results = run_3d_editing(
        pipeline, render_dir, image_dir, output_path, use_text, src_prompt, tgt_prompt, **(edit_params or {})
    )
    results["editing"] = edit_results

    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Final result: {output_path}")

    return results


def main():
    """
    Main function for command line usage
    """
    parser = argparse.ArgumentParser(description="Complete 3D Editing Pipeline")

    # Required arguments
    parser.add_argument(
        "--input_model", type=str, required=True, help="Path to input 3D model file"
    )
    parser.add_argument(
        "--mask_glb",
        type=str,
        required=True,
        help="Path to mask GLB file (defines the region to be edited)",
    )
    parser.add_argument(
        "--render_dir", type=str, required=True, help="Directory for render outputs"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing source, target, and mask images",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path for final output GLB file (must end with .glb)",
    )

    # Optional rendering parameters
    parser.add_argument(
        "--num_views", type=int, default=150, help="Number of views to render"
    )
    parser.add_argument(
        "--resolution", type=int, default=512, help="Rendering resolution"
    )
    parser.add_argument(
        "--render_engine", type=str, default="CYCLES", help="Rendering engine"
    )

    # Optional feature extraction parameters
    parser.add_argument(
        "--feature_model",
        type=str,
        default="dinov2_vitl14_reg",
        help="Feature extraction model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for feature extraction"
    )

    # Optional masking parameters
    parser.add_argument(
        "--filter_method", type=str, default="volume", help="Voxel filtering method"
    )
    parser.add_argument(
        "--voxel_size", type=float, default=1 / 64, help="Voxel size for masking"
    )

    # Optional editing parameters
    parser.add_argument(
        "--skip_step", type=int, default=0, help="Skip steps in editing"
    )
    parser.add_argument(
        "--re_init", action="store_true", help="Reinitialize during editing"
    )
    parser.add_argument(
        "--cfg_strength",
        type=float,
        nargs=4,
        default=[5.0, 6.0, 0.0, 0.0],
        help="CFG strength parameters [stage1_inv, stage1_fwd, stage2_inv, stage2_fwd]",
    )

    args = parser.parse_args()

    # Prepare parameters
    render_params = {
        "num_views": args.num_views,
        "resolution": args.resolution,
        "engine": args.render_engine,
    }

    feature_params = {"model": args.feature_model, "batch_size": args.batch_size}

    mask_params = {"filter_method": args.filter_method, "voxel_size": args.voxel_size}

    edit_params = {
        "skip_step": args.skip_step,
        "re_init": args.re_init,
        "cfg": args.cfg_strength,
    }

    # Run pipeline
    try:
        pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large"
        )
        pipeline.cuda()
        results = run_complete_pipeline(
            pipeline=pipeline,
            input_model_path=args.input_model,
            mask_glb_path=args.mask_glb,
            render_dir=args.render_dir,
            image_dir=args.image_dir,
            output_path=args.output_path,
            use_text=args.use_text,
            src_prompt=args.src_prompt,
            tgt_prompt=args.tgt_prompt,
            render_params=render_params,
            feature_params=feature_params,
            mask_params=mask_params,
            edit_params=edit_params,
        )
        print("Pipeline completed successfully!")
        return results
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, default="assets/example/model.glb")
    parser.add_argument("--mask_model", type=str, default="assets/example/mask.glb")
    parser.add_argument("--image_dir", type=str, default="assets/example/images")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--render_dir",
        type=str,
        default=None,
        help="Skip rendering and use existing render directory",
    )
    parser.add_argument("--use_text", default=False, action="store_true", help="Use text for editing")
    parser.add_argument("--src_prompt", type=str, default=PROMPT_SRC, help="Source prompt for text editing")
    parser.add_argument("--tgt_prompt", type=str, default=PROMPT_3D_EDIT, help="Target prompt for text editing")
    args = parser.parse_args()

    if args.use_text:
        pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-large")
    else:
        pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    run_complete_pipeline(
        pipeline=pipeline,
        input_model_path=args.input_model,
        mask_glb_path=args.mask_model,
        render_dir=(
            args.render_dir
            if args.render_dir
            else os.path.join(args.output_dir, "render")
        ),
        image_dir=args.image_dir,
        output_path=os.path.join(args.output_dir, "output.glb"),
        use_text=args.use_text,
        src_prompt=args.src_prompt,
        tgt_prompt=args.tgt_prompt,
    )

    print(
        f"Pipeline completed successfully! Result saved to `output.glb` in `{args.output_dir}`"
    )
