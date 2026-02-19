#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

def norm_caption(s: str) -> str:
    # normalize: strip, collapse whitespace, lowercase
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def load_captions_txt(txt_path: Path) -> set[str]:
    caps = set()
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            caps.add(norm_caption(line))
    return caps

def filter_csv_by_captions(in_csv: Path, captions_set: set[str], out_csv: Path) -> int:
    matched = 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Use csv module to correctly handle commas/quotes in captions
    with in_csv.open("r", encoding="utf-8", newline="") as fin, \
         out_csv.open("w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        for row in reader:
            # Expecting rows like:
            # id, split, caption, image_id, seed
            if len(row) < 3:
                continue

            caption = row[2]
            if norm_caption(caption) in captions_set:
                writer.writerow(row)
                matched += 1

    return matched

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions_txt", required=True, help="Path to txt file with one caption per line")
    ap.add_argument("--input_csv", required=True, help="Path to original CSV")
    ap.add_argument("--output_csv", required=True, help="Path to write filtered CSV")
    args = ap.parse_args()

    captions = load_captions_txt(Path(args.captions_txt))
    n = filter_csv_by_captions(Path(args.input_csv), captions, Path(args.output_csv))

    print(f"Loaded {len(captions)} captions from txt")
    print(f"Wrote {n} matching rows to: {args.output_csv}")

if __name__ == "__main__":
    main()
