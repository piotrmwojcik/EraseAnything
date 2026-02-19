#!/usr/bin/env python3
import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import time
import re
import shutil
from pathlib import Path

from diffusers import FluxPipeline

# -----------------------------------
# Prompt cleanup
# -----------------------------------
def coerce_prompt(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""

    if isinstance(v, (list, tuple, set)):
        return ", ".join(str(x).strip() for x in v if str(x).strip())

    s = str(v).strip()
    s = re.sub(r'^\s*prompt\s*[:\-]?\s*', "", s, flags=re.I)

    m = re.match(r'^\[\s*(.*)\s*\]$', s)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        return ", ".join(p for p in parts if p)

    return s

def clean_fname(v) -> str:
    # Removes Windows CR and surrounding whitespace
    return str(v).strip().replace("\r", "")

# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="generated_flux")
    parser.add_argument("--coco_out_dir", type=str, required=True,
                        help="Where to copy the corresponding COCO GT images for comparison")
    parser.add_argument("--coco_dir", type=str, required=True,
                        help="Directory that contains COCO images (e.g., .../val2014)")

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--device", type=str, default="cuda")

    # column names (so it works with different CSV schemas)
    parser.add_argument("--prompt_col", type=str, default="prompt")
    parser.add_argument("--seed_col", type=str, default="evaluation_seed")
    parser.add_argument("--coco_file_col", type=str, default="coco_file",
                        help="Column that contains COCO filename like COCO_val2014_000000278829.jpg")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.coco_out_dir, exist_ok=True)

    # -----------------------------------
    # Load pipeline
    # -----------------------------------
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # -----------------------------------
    # Load CSV
    # -----------------------------------
    # Keep it robust to weird encodings/CRLF
    df = pd.read_csv(args.csv_path, encoding="utf-8", engine="python")

    # If user has no header, allow fallback to positional columns:
    # e.g. (id, coco_filename, caption/prompt)
    if args.coco_file_col not in df.columns and len(df.columns) >= 2:
        # assume second column is coco filename if not found
        df = df.rename(columns={df.columns[1]: args.coco_file_col})
    if args.prompt_col not in df.columns and len(df.columns) >= 3:
        # assume third column is prompt/caption if not found
        df = df.rename(columns={df.columns[2]: args.prompt_col})

    # -----------------------------------
    # Generate + copy GT
    # -----------------------------------
    for idx, row in tqdm(df.iterrows(), total=len(df)):

        prompt = coerce_prompt(row.get(args.prompt_col, ""))
        if not prompt:
            continue

        seed_val = row.get(args.seed_col, 0)
        try:
            seed = int(str(seed_val).strip().replace("\r", ""))
        except Exception:
            seed = 0

        generator = torch.Generator(device=str(device)).manual_seed(seed)

        # Generated image path
        gen_name = f"{idx:05d}.png"
        image_path = os.path.join(args.output_dir, gen_name)

        # COCO file copy path (match index for easy side-by-side)
        coco_fname = clean_fname(row.get(args.coco_file_col, ""))
        coco_src = os.path.join(args.coco_dir, coco_fname) if coco_fname else ""
        coco_dst = os.path.join(args.coco_out_dir, f"{idx:05d}{Path(coco_fname).suffix or '.jpg'}")

        # Skip if both already exist
        if os.path.exists(image_path) and os.path.exists(coco_dst):
            continue

        start = time.time()

        # Generate if missing
        if not os.path.exists(image_path):
            img = pipe(
                prompt=prompt,
                generator=generator,
                guidance_scale=args.guidance_scale,
                height=args.image_size,
                width=args.image_size,
                num_inference_steps=args.num_inference_steps,
                max_sequence_length=256
            ).images[0]
            img.save(image_path)

        # Copy COCO GT if missing
        if coco_fname:
            if os.path.exists(coco_src):
                if not os.path.exists(coco_dst):
                    shutil.copy2(coco_src, coco_dst)
            else:
                print(f"[{idx}] WARNING: COCO file not found: {coco_src}")
        else:
            print(f"[{idx}] WARNING: No COCO filename in column '{args.coco_file_col}'")

        end = time.time()
        print(f"[{idx}] Seed={seed} | {end - start:.2f}s | gen={image_path} | coco={coco_dst}")
