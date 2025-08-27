# Adapted from https://github.com/huanngzh/bpy-renderer
import json
import os
import shutil
from glob import glob

import cv2
import numpy as np
from bpyrenderer import SceneManager
from bpyrenderer.camera import add_camera
from bpyrenderer.camera.layout import get_camera_positions_on_sphere
from bpyrenderer.engine import init_render_engine
from bpyrenderer.environment import set_env_map
from bpyrenderer.importer import load_file
from bpyrenderer.render_output import enable_color_output, enable_depth_output

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def render_model(input_model, output_dir):
    # 1. Init engine and scene manager
    # init_render_engine("BLENDER_EEVEE")
    init_render_engine("CYCLES")
    scene_manager = SceneManager()
    scene_manager.clear(reset_keyframes=True)

    # 2. Import models
    load_file(input_model)

    # Others. smooth objects and normalize scene
    scene_manager.smooth()
    scene_manager.clear_normal_map()
    scene_manager.set_material_transparency(False)
    scene_manager.set_materials_opaque()  # !!! Important for render normal but may cause render error !!!
    # scene_manager.normalize_scene(1.0) # Not normalize here

    # 3. Set environment
    set_env_map("assets/preset/brown_photostudio_02_1k.exr")
    # set_background_color([1.0, 1.0, 1.0, 1.0])

    # 4. Prepare cameras
    cam_pos, cam_mats, elevations, azimuths = get_camera_positions_on_sphere(
        center=(0, 0, 0),
        radius=1.8,
        elevations=[15],
        azimuths=[item - 90 for item in [90, 45, 0, 315, 270]],
    )
    cameras = []
    for i, camera_mat in enumerate(cam_mats):
        camera = add_camera(camera_mat, "PERSP", add_frame=i < len(cam_mats) - 1)
        cameras.append(camera)

    # 5. Set render outputs
    width, height = 1024, 1024
    enable_color_output(
        width,
        height,
        output_dir,
        file_format="PNG",
        mode="IMAGE",
        film_transparent=True,
    )
    enable_depth_output(output_dir)
    scene_manager.render()

    # Optional. save metadata
    meta_info = {"width": width, "height": height, "locations": []}
    for i in range(len(cam_pos)):
        index = "{0:04d}".format(i)
        meta_info["locations"].append(
            {
                "index": index,
                "projection_type": cameras[i].data.type,
                "ortho_scale": cameras[i].data.ortho_scale,
                "camera_angle_x": cameras[i].data.angle_x,
                "elevation": elevations[i],
                "azimuth": azimuths[i],
                "transform_matrix": cam_mats[i].tolist(),
            }
        )
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta_info, f, indent=4)


def load_depth(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[..., 0]
    mask = ~(depth > 1000.0)
    depth[mask < 0.5] = 10000.0  # background depth set to infity

    return depth, mask


def process(input_model, input_mask, output_dir):
    model_dir = os.path.join(output_dir, "rgb")
    mask_dir = os.path.join(output_dir, "mask_3d")

    render_model(input_model, model_dir)
    render_model(input_mask, mask_dir)

    render_dir = os.path.join(output_dir, "images")
    # NOTE!!!
    if os.path.exists(render_dir):
        shutil.rmtree(render_dir)
    os.makedirs(render_dir, exist_ok=True)

    model_depths = sorted(glob(os.path.join(model_dir, "*.exr")))
    mask_depths = sorted(glob(os.path.join(mask_dir, "*.exr")))

    if len(model_depths) != len(mask_depths):
        raise ValueError("Number of depth files must be the same!")

    for i in range(len(model_depths)):
        model_depth, model_mask = load_depth(model_depths[i])
        mask_depth, mask_mask = load_depth(mask_depths[i])

        # NOTE: signed (consider occlusion)
        diff_mask = (model_depth - mask_depth > 1e-4).astype(np.uint8)
        diff_mask[mask_mask <= 0.5] = 0

        out_path = os.path.join(render_dir, f"mask_{i:04d}.png")
        cv2.imwrite(out_path, diff_mask * 255)

    model_rgbs = sorted(glob(os.path.join(model_dir, "*.png"))) + [
        os.path.join(model_dir, "meta.json")
    ]
    for i, file in enumerate(model_rgbs):
        shutil.copy(file, os.path.join(render_dir, file.split("/")[-1]))

        if not file.endswith("meta.json"):
            rgba_img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

            if rgba_img is not None and rgba_img.shape[-1] == 4:
                # split rgb and alpha
                rgb = rgba_img[..., :3]
                alpha = rgba_img[..., 3] / 255.0

                # 1. convert to white background
                white_bg = np.ones_like(rgb) * 255
                rgb_with_white_bg = rgb * alpha[..., np.newaxis] + white_bg * (
                    1 - alpha[..., np.newaxis]
                )

                # 2. apply mask
                filename = file.split("/")[-1]
                if filename.startswith("render_"):
                    index = filename.split("_")[-1].split(".")[0]
                    mask_file = os.path.join(render_dir, f"mask_{index}.png")
                else:
                    mask_file = None
                if mask_file and os.path.exists(mask_file):
                    mask = 1 - cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) / 255.0
                    rgb_with_white_bg = rgb_with_white_bg * mask[..., np.newaxis]

                output_file = os.path.join(render_dir, f"visual_{i:04}.png")
                cv2.imwrite(output_file, rgb_with_white_bg.astype(np.uint8))

    shutil.rmtree(model_dir)
    shutil.rmtree(mask_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model", type=str, default="assets/example/model.glb")
    parser.add_argument("--mask_model", type=str, default="assets/example/mask.glb")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    source_model = args.source_model
    mask_model = args.mask_model
    output_dir = os.path.join(args.output_dir)

    process(source_model, mask_model, output_dir)

    # copy input models to output_dir
    shutil.copy(source_model, os.path.join(args.output_dir, "model.glb"))
    shutil.copy(mask_model, os.path.join(args.output_dir, "mask.glb"))

    print(
        f"Rendering completed! Result saved to {args.output_dir}, including (input) model.glb, mask.glb, (output) images."
    )
    print(
        f"Now you can selete one pair including `render_xxxx.png` and `mask_xxxx.png` in the `images` directory to inpaint."
    )
