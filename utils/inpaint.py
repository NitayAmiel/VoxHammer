import torch
from diffusers import FluxFillPipeline
from PIL import Image


def inpaint(
    pipeline,
    image_path,
    mask_path,
    output_path,
    prompt,
    height=1024,
    width=1024,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
    seed=0,
):
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    if image.mode == "RGBA":
        # convert to RGB with white background
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.paste(image, (0, 0), image)
        image = background.convert("RGB")

    image = pipeline(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    image.save(output_path)


if __name__ == "__main__":
    import argparse
    import os
    import shutil

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path", type=str, default="example/images/2d_render.png"
    )
    parser.add_argument("--mask_path", type=str, default="example/images/2d_mask.png")
    parser.add_argument("--output_dir", type=str, default="outputs/images")
    parser.add_argument("--prompt", type=str, default="A dog.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load pipeline
    pipeline = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16
    ).to("cuda")

    # inpaint
    inpaint(
        pipeline=pipeline,
        image_path=args.image_path,
        mask_path=args.mask_path,
        output_path=os.path.join(args.output_dir, "2d_edit.png"),
        prompt=args.prompt,
    )

    # copy input images to output_dir
    shutil.copy(args.image_path, os.path.join(args.output_dir, "2d_render.png"))
    shutil.copy(args.mask_path, os.path.join(args.output_dir, "2d_mask.png"))

    print(
        f"Inpainting completed! Result saved to {args.output_dir}, including (input) 2d_render.png, 2d_mask.png, (output) 2d_edit.png"
    )
