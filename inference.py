#!/usr/bin/env python3

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import torch
import pandas as pd
from tqdm import tqdm
import time
import re

from diffusers import FluxPipeline
from huggingface_hub import login

# -----------------------------------
# Login (optional)
# -----------------------------------

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


# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="generated_flux")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lora_path", type=str, default="Flux-erase-dev/pytorch_lora_weights.safetensors")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -----------------------------------
    # Load pipeline
    # -----------------------------------
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # -----------------------------------
    # Load CSV
    # -----------------------------------
    df = pd.read_csv(args.csv_path)

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------
    # Generate
    # -----------------------------------
    for idx, row in tqdm(df.iterrows(), total=len(df)):

        prompt = coerce_prompt(row.get("prompt", ""))
        if not prompt:
            continue

        seed = int(row.get("evaluation_seed", 0))
        generator = torch.Generator(device=device).manual_seed(seed)

        image_path = os.path.join(args.output_dir, f"{idx:05d}.png")
        if os.path.exists(image_path):
            continue

        start = time.time()

        # 3️⃣ Call pipeline with embeds
        image = pipe(prompt=prompt,
                     # generator=generator,
                     guidance_scale=3.0,
                     height=512,
                     width=512,
                     num_inference_steps=28,
                     max_sequence_length=256).images[0]

        image.save(image_path)

        end = time.time()
        print(f"[{idx}] Seed={seed} | {end - start:.2f}s | {image_path}")
