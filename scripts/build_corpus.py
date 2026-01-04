#!/usr/bin/env python3
"""
build_corpus.py

Build unique_titles corpus from a pairs dataframe (title1/title2), assign stable ids.

Input:
  - CSV or Parquet with columns: title1, title2 (label optional)

Output:
  - CSV/Parquet with columns: id, title
"""

from __future__ import annotations

import argparse
import hashlib
import os
from typing import Optional

import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet"]:
        return pd.read_parquet(path)
    if ext in [".csv"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {ext}. Use .csv or .parquet")


def write_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet"]:
        df.to_parquet(path, index=False)
        return
    if ext in [".csv"]:
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {ext}. Use .csv or .parquet")


def normalize_text(s: str) -> str:
    # Light normalization: trim + collapse spaces
    return " ".join(str(s).strip().split())


def stable_id_from_text(text: str, prefix: str = "t_") -> str:
    # Stable 64-bit-ish id from text content
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return f"{prefix}{h}"


def build_unique_titles(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    lowercase: bool = False,
) -> pd.DataFrame:
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Missing required columns: {col1}, {col2}")

    s1 = df[col1].dropna().astype(str).map(normalize_text)
    s2 = df[col2].dropna().astype(str).map(normalize_text)

    if lowercase:
        s1 = s1.str.lower()
        s2 = s2.str.lower()

    all_titles = pd.concat([s1, s2], ignore_index=True)
    all_titles = all_titles[all_titles.str.len() > 0]

    unique_titles = all_titles.drop_duplicates().reset_index(drop=True)

    out = pd.DataFrame({"title": unique_titles})
    out["id"] = out["title"].map(stable_id_from_text)
    out = out[["id", "title"]]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to pairs dataset (.csv/.parquet)")
    p.add_argument("--output", required=True, help="Path to corpus output (.csv/.parquet)")
    p.add_argument("--col1", default="title1", help="First title column")
    p.add_argument("--col2", default="title2", help="Second title column")
    p.add_argument("--lowercase", action="store_true", help="Lowercase titles before dedup")
    args = p.parse_args()

    df = read_table(args.input)
    corpus = build_unique_titles(df, args.col1, args.col2, lowercase=args.lowercase)

    print(f"[build_corpus] pairs rows: {len(df):,}")
    print(f"[build_corpus] unique titles: {len(corpus):,}")
    write_table(corpus, args.output)
    print(f"[build_corpus] saved -> {args.output}")


if __name__ == "__main__":
    main()