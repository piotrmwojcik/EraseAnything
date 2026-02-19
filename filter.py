#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

def norm_caption(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def load_captions_txt(txt_path: Path) -> list[str]:
    captions = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                captions.append(norm_caption(line))
    return captions

def build_caption_lookup(in_csv: Path) -> dict[str, list[str]]:
    lookup = {}
    with in_csv.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.reader(fin)
        for row in reader:
            if len(row) < 3:
                continue
            lookup[norm_caption(row[2])] = row
    return lookup

def filter_csv_by_captions(in_csv: Path, captions_list: list[str], out_csv: Path) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    lookup = build_caption_lookup(in_csv)
    matched = 0

    with out_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)

        for cap in captions_list:
            if cap in lookup:
                writer.writerow(lookup[cap])
                matched += 1
            else:
                print(f"Warning: caption not found -> {cap}")

    return matched

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions_txt", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    captions = load_captions_txt(Path(args.captions_txt))
    n = filter_csv_by_captions(Path(args.input_csv), captions, Path(args.output_csv))

    print(f"Loaded {len(captions)} captions from txt")
    print(f"Wrote {n} matching rows to: {args.output_csv}")

if __name__ == "__main__":
    main()
